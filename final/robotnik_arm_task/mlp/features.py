"""Ordered MLP feature schemas and checkpoint policy metadata."""

import math

import numpy as np

INPUTS = (['base_x', 'base_y', 'base_yaw']
          + [f'arm_q{i}' for i in range(1, 8)]
          + [f'arm_qd{i}' for i in range(1, 8)]
          + ['target_x', 'target_y', 'target_z', 'base_vx', 'base_vy',
             'base_yaw_rate', 'base_ax', 'base_ay', 'base_yaw_acc']
          + [f'arm_qdd{i}' for i in range(1, 8)])
OUTPUTS = [f'wheel_cmd{i}' for i in range(1, 5)] + [f'arm_torque{i}' for i in range(1, 8)]
HEADER = ['trajectory_id', 'time', 'phase'] + INPUTS + OUTPUTS
GOAL_INPUTS = INPUTS + ['distance_to_base_goal', 'signed_heading_error_to_base_goal']
TARGET_INPUTS = INPUTS + ['relative_target_x_body','relative_target_y_body',
                         'distance_to_target_xy','heading_error_to_target']
TARGET_FEATURE_POLICY = ('target37_v1: baseline inputs cast to float32 first; geometry in float64; '
    'dx=target_x-base_x, dy=target_y-base_y, psi=base_yaw; '
    'body_x=cos(psi)*dx+sin(psi)*dy; body_y=-sin(psi)*dx+cos(psi)*dy; '
    'distance=hypot(dx,dy); error=atan2(dy,dx)-psi; heading=atan2(sin(error),cos(error)); '
    'heading=0 when dx=dy=0 after float32 input casting; append four float32 features.')
TARGET37_MOTION_NAMES = ['base_vx','base_vy','base_yaw_rate','base_ax','base_ay','base_yaw_acc']
TARGET37_FILTER_POLICY = dict(version=1,type='two cascaded exponential pose smoothers, then backward differences',
    dt_s=.001,tau_per_stage_s=.020,alpha_formula='1-exp(-dt/tau)',stages=2,
    pose_features=['base_x','base_y','base_yaw'],feature_names=TARGET37_MOTION_NAMES.copy(),
    indices=[TARGET_INPUTS.index(name) for name in TARGET37_MOTION_NAMES],index_base=0,
    angular_policy='accumulate atan2(sin(yaw-prev_yaw),cos(yaw-prev_yaw)) before smoothing',
    startup='seed both stages with first measured pose; first velocity and acceleration zero',
    reset='each independent trajectory/episode; preserve state across CSV chunks',
    order='float32 observations -> float64 pose estimation -> replace six float32 motion slots -> training-fitted input scaling; geometry and arm features unchanged')

HISTORY_INPUTS = ['current/'+name for name in GOAL_INPUTS]+['lag100ms/'+name for name in GOAL_INPUTS]
HISTORY_POLICY = dict(lag_steps=100,sampling_interval_s=.001,
    padding='repeat_first_observation',block_order=['current','lag100ms'],
    observation_features='goal35',reset='each trajectory/episode',version=1)
GOAL_FEATURE_POLICY = ('goal35_v1: baseline observations rounded to float32 first; geometry calculated in float64, '
    'then appended as float32. Production height-dependent standoff and origin-based target bearing; '
    'heading wrapped with atan2(sin(error),cos(error)); atan2(0,0)=0 convention.')


class CausalBaseFilter:
    """Fixed 20ms-per-stage pose estimator; no future samples or raw-derivative filtering."""
    def __init__(self):
        self.count = 0
        self.first = self.second = self.velocity = None
        self.previous_yaw = self.unwrapped_yaw = None
        self.alpha = -math.expm1(-TARGET37_FILTER_POLICY['dt_s']/TARGET37_FILTER_POLICY['tau_per_stage_s'])

    def build(self, observations, start_step=None):
        if start_step is not None and start_step!=self.count:
            raise ValueError('Filtered trajectory must start at zero and remain contiguous')
        result = model_features(observations,'target37')
        pose_indices = [TARGET_INPUTS.index(name) for name in TARGET37_FILTER_POLICY['pose_features']]
        dt = TARGET37_FILTER_POLICY['dt_s']
        for row in result:
            pose = row[pose_indices].astype(np.float64)
            yaw = float(pose[2])
            if self.count==0:
                self.unwrapped_yaw = yaw
                self.first,self.second = pose.copy(),pose.copy()
                velocity,acceleration = np.zeros(3),np.zeros(3)
            else:
                delta = yaw-self.previous_yaw
                self.unwrapped_yaw += math.atan2(math.sin(delta),math.cos(delta))
                pose[2] = self.unwrapped_yaw
                self.first += self.alpha*(pose-self.first)
                smooth = self.second+self.alpha*(self.first-self.second)
                velocity = (smooth-self.second)/dt
                acceleration = (velocity-self.velocity)/dt
                self.second = smooth
            row[TARGET37_FILTER_POLICY['indices']] = np.r_[velocity,acceleration].astype(np.float32)
            self.velocity,self.previous_yaw = velocity,yaw
            self.count += 1
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite filtered features')
        return result


class History100:
    """Causal Goal35 history shared by CSV chunks and single-step inference."""
    def __init__(self):
        self.tail = np.empty((0,35),dtype=np.float32)
        self.first = None
        self.count = 0

    def build(self, current, start_step=None):
        current = np.asarray(current,dtype=np.float32)
        if current.ndim!=2 or current.shape[1]!=35 or not np.isfinite(current).all():
            raise ValueError('History expects finite chronological N-by-35 Goal35 observations')
        if start_step is not None and start_step!=self.count:
            raise ValueError('History must start at step zero and remain contiguous')
        if not len(current):
            return np.empty((0,70),dtype=np.float32)
        if self.first is None:
            self.first = current[0].copy()
        combined = np.concatenate((self.tail,current),axis=0)
        indices = np.arange(len(current))+len(self.tail)-100
        lagged = combined[np.maximum(indices,0)].copy()
        lagged[indices<0] = self.first
        result = np.concatenate((current,lagged),axis=1)
        self.tail = combined[-100:].copy()
        self.count += len(current)
        return result


def model_features(observations, feature_set='baseline33', base_filter=None, start_step=None):
    """Shared row/inference transformation; accepts original ordered N-by-33 inputs."""
    x = np.asarray(observations,dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 33 or not np.isfinite(x).all():
        raise ValueError('Expected finite N-by-33 baseline observations')
    if feature_set == 'baseline33':
        return x
    if feature_set == 'target37_filtered':
        if not isinstance(base_filter,CausalBaseFilter):
            raise ValueError('Target37 filtered requires a persistent CausalBaseFilter, reset each trajectory')
        return base_filter.build(x,start_step)
    if feature_set=='target37':
        dx = x[:,17].astype(np.float64)-x[:,0].astype(np.float64)
        dy = x[:,18].astype(np.float64)-x[:,1].astype(np.float64)
        psi = x[:,2].astype(np.float64)
        c,s = np.cos(psi),np.sin(psi)
        error = np.arctan2(dy,dx)-psi
        heading = np.arctan2(np.sin(error),np.cos(error))
        heading = np.where((dx==0)&(dy==0),0.,heading)
        extra = np.column_stack((c*dx+s*dy,-s*dx+c*dy,np.hypot(dx,dy),heading))
        result = np.concatenate((x,extra.astype(np.float32)),axis=1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite target-relative features')
        return result
    if feature_set != 'goal35':
        raise ValueError('Unsupported feature set')
    target = x[:,17:20].astype(np.float64)
    standoff = np.interp(target[:,2],[.1,.5,1.1,1.25],[.8,.65,.65,.35])
    bearing = np.where(np.linalg.norm(target[:,:2],axis=1)>1e-4,
                       np.arctan2(target[:,1],target[:,0]),0.)
    goal_x = target[:,0]-standoff*np.cos(bearing)
    goal_y = target[:,1]-standoff*np.sin(bearing)
    dx,dy = goal_x-x[:,0].astype(float),goal_y-x[:,1].astype(float)
    error = np.arctan2(dy,dx)-x[:,2].astype(float)
    extra = np.column_stack((np.hypot(dx,dy),np.arctan2(np.sin(error),np.cos(error))))
    result = np.concatenate((x,extra.astype(np.float32)),axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite goal-relative features')
    return result

def transform(x, scaler, feature_set='baseline33'):
    result = x if scaler is None else scaler.transform(x).astype(np.float32, copy=False)
    return result


