# LMP 관련 모듈 및 유틸리티 불러오기
from LMP import LMP
from utils import get_clock_time, normalize_vector, pointat2quat, bcolors, Observation, VoxelIndexingWrapper # 벡터->쿼터니언회전행렬
import numpy as np
from planners import PathPlanner
import time
from scipy.ndimage import distance_transform_edt
import transforms3d # 좌표변환 라이브러리
from controllers import Controller

# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
# LLM이 생성하는 용어 통일
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']


# LMP와 환경을 연결하는 인터페이스
class LMP_interface():

  def __init__(self, env, lmp_config, controller_config, planner_config, env_name='rlbench'):
    self._env = env # 로봇환경객체
    self._env_name = env_name # 환경 이름 rlbench
    self._cfg = lmp_config # LMP 모델 설정값
    self._map_size = self._cfg['map_size'] # 맵 크기 저장
    self._planner = PathPlanner(planner_config, map_size=self._map_size) # 플래너초기화
    self._controller = Controller(self._env, controller_config) # 컨트롤러초기화

    # calculate size of each voxel (resolution)
    # 작업공간 크기 기반의 voxel resolution 계산.
    self._resolution = (self._env.workspace_bounds_max - self._env.workspace_bounds_min) / self._map_size
    print('#' * 50)
    print(f'## voxel resolution: {self._resolution}')
    print('#' * 50)
    print()
    print()
  
  # ======================================================
  # == functions exposed to LLM, LLM이 사용하도록 제공할 함수들.
  # ======================================================
  
  # 로봇 ee 위치를 반환.
  def get_ee_pos(self):
    return self._world_to_voxel(self._env.get_ee_pos())
  
  # 특정 obj_name의 객체를 탐지하고, 해당 객체 정보를 반환.
  def detect(self, obj_name):
    """return an observation dict containing useful information about the object"""
    if obj_name.lower() in EE_ALIAS: # ee 탐지의 경우.
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self.get_ee_pos()
      obs_dict['aabb'] = np.array([self.get_ee_pos(), self.get_ee_pos()])
      obs_dict['_position_world'] = self._env.get_ee_pos()
    elif obj_name.lower() in TABLE_ALIAS: # table 탐지의 경우.
      offset_percentage = 0.1
      x_min = self._env.workspace_bounds_min[0] + offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      x_max = self._env.workspace_bounds_max[0] - offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      y_min = self._env.workspace_bounds_min[1] + offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      y_max = self._env.workspace_bounds_max[1] - offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      table_max_world = np.array([x_max, y_max, 0])
      table_min_world = np.array([x_min, y_min, 0])
      table_center = (table_max_world + table_min_world) / 2
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self._world_to_voxel(table_center)
      obs_dict['_position_world'] = table_center
      obs_dict['normal'] = np.array([0, 0, 1])
      obs_dict['aabb'] = np.array([self._world_to_voxel(table_min_world), self._world_to_voxel(table_max_world)])
    else: # 일반 객체 탐지의 경우.
      obs_dict = dict()
      obj_pc, obj_normal = self._env.get_3d_obs_by_name(obj_name) # 3d pc 및 법선벡터정보.
      voxel_map = self._points_to_voxel_map(obj_pc) # 3d pc를 복셀로 변환.
      aabb_min = self._world_to_voxel(np.min(obj_pc, axis=0))
      aabb_max = self._world_to_voxel(np.max(obj_pc, axis=0))
      obs_dict['occupancy_map'] = voxel_map  # in voxel frame
      obs_dict['name'] = obj_name
      obs_dict['position'] = self._world_to_voxel(np.mean(obj_pc, axis=0))  # in voxel frame
      obs_dict['aabb'] = np.array([aabb_min, aabb_max])  # in voxel frame
      obs_dict['_position_world'] = np.mean(obj_pc, axis=0)  # in world frame
      obs_dict['_point_cloud_world'] = obj_pc  # in world frame
      obs_dict['normal'] = normalize_vector(obj_normal.mean(axis=0))

    object_obs = Observation(obs_dict) # Observation 객체로 변환 후 반환.
    return object_obs
  

  # 로봇이 특정 물체를 이동시키기 위한 경로를 계획하고 실행.
  ## 경로 계획 - 경로 실행 - 실시간 피드백 반영
  def execute(self, movable_obs_func, affordance_map=None, avoidance_map=None, rotation_map=None,
              velocity_map=None, gripper_map=None):
    """
    First use planner to generate waypoint path, then use controller to follow the waypoints.

    Args:
      movable_obs_func: callable function to get observation of the body to be moved # 이동할 객체의 정보.
      affordance_map: callable function that generates a 3D numpy array, the target voxel map
      avoidance_map: callable function that generates a 3D numpy array, the obstacle voxel map
      rotation_map: callable function that generates a 4D numpy array, the rotation voxel map (rotation is represented by a quaternion *in world frame*)
      velocity_map: callable function that generates a 3D numpy array, the velocity voxel map
      gripper_map: callable function that generates a 3D numpy array, the gripper voxel map
    """
    # initialize default voxel maps if not specified # 기본값 설정.
    if rotation_map is None:
      rotation_map = self._get_default_voxel_map('rotation')
    if velocity_map is None:
      velocity_map = self._get_default_voxel_map('velocity')
    if gripper_map is None:
      gripper_map = self._get_default_voxel_map('gripper')
    if avoidance_map is None:
      avoidance_map = self._get_default_voxel_map('obstacle')
    
    # 로봇이 ee가 아닌, 다른 객체를 이동하는지 확인 변수.
    object_centric = (not movable_obs_func()['name'] in EE_ALIAS)
    execute_info = []

    # 목표(affordance_map)이 제공된 경우 이동 시작.
    if affordance_map is not None:
      # execute path in closed-loop
      for plan_iter in range(self._cfg['max_plan_iter']): # 최대 시도횟수만큼 반복(실패 시 재계획 가능)
        step_info = dict()

        # 최신 정보 업데이트 (실시간 감지)
        # evaluate voxel maps such that we use latest information
        movable_obs = movable_obs_func()
        _affordance_map = affordance_map()
        _avoidance_map = avoidance_map()
        _rotation_map = rotation_map()
        _velocity_map = velocity_map()
        _gripper_map = gripper_map()

        # 장애물 맵 전처리 (경로 계획에 반영)
        _avoidance_map = self._preprocess_avoidance_map(_avoidance_map, _affordance_map, movable_obs)

        # 경로 계획 시작
        start_pos = movable_obs['position']
        start_time = time.time()

        # optimize path and log
        path_voxel, planner_info = self._planner.optimize(start_pos, _affordance_map, _avoidance_map,
                                                        object_centric=object_centric)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] planner time: {time.time() - start_time:.3f}s{bcolors.ENDC}')
        assert len(path_voxel) > 0, 'path_voxel is empty'

        # 계획된 경로를 저장
        step_info['path_voxel'] = path_voxel
        step_info['planner_info'] = planner_info

        # Voxel 좌표 경로를 World 좌표 경로로 변환.
        # convert voxel path to world trajectory, and include rotation, velocity, and gripper information
        traj_world = self._path2traj(path_voxel, _rotation_map, _velocity_map, _gripper_map)
        traj_world = traj_world[:self._cfg['num_waypoints_per_plan']]
        step_info['start_pos'] = start_pos
        step_info['plan_iter'] = plan_iter
        step_info['movable_obs'] = movable_obs
        step_info['traj_world'] = traj_world
        step_info['affordance_map'] = _affordance_map
        step_info['rotation_map'] = _rotation_map
        step_info['velocity_map'] = _velocity_map
        step_info['gripper_map'] = _gripper_map
        step_info['avoidance_map'] = _avoidance_map

        # 시각화(옵션)
        if self._cfg['visualize']:
          assert self._env.visualizer is not None
          step_info['start_pos_world'] = self._voxel_to_world(start_pos)
          step_info['targets_world'] = self._voxel_to_world(planner_info['targets_voxel'])
          self._env.visualizer.visualize(step_info)

        # 계획된 경로 실행.
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] start executing path via controller ({len(traj_world)} waypoints){bcolors.ENDC}')
        controller_infos = dict()
        for i, waypoint in enumerate(traj_world):
          # 이동이 완료되었는지 확인 (마지막 지점과 거리가 0.01M 이하인 경우)
          if np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0]) <= 0.01:
            print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={movable_obs['_position_world']}, target={traj_world[-1][0]} (distance: {np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0]):.3f})){bcolors.ENDC}")
            break
          # skip waypoint if moving to this point is going in opposite direction of the final target point
          # (for example, if you have over-pushed an object, no need to move back)
          # 역방향으로 이동하는 경우 해당 웨이포인트 건너뛰기 (불필요한 이동 방지)
          if i != 0 and i != len(traj_world) - 1:
            movable2target = traj_world[-1][0] - movable_obs['_position_world']
            movable2waypoint = waypoint[0] - movable_obs['_position_world']
            if np.dot(movable2target, movable2waypoint).round(3) <= 0:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] skip waypoint {i+1} because it is moving in opposite direction of the final target{bcolors.ENDC}')
              continue

          # 컨트롤러 사용하여 해당 웨이포인트로 이동.
          controller_info = self._controller.execute(movable_obs, waypoint)
          # 이동 후 현재 위치 업데이트
          movable_obs = movable_obs_func()
          dist2target = np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0])
          if not object_centric and controller_info['mp_info'] == -1:
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] failed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}); mp info: {controller_info["mp_info"]}{bcolors.ENDC}')
          else:
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}){bcolors.ENDC}')
          controller_info['controller_step'] = i
          controller_info['target_waypoint'] = waypoint
          controller_infos[i] = controller_info


        step_info['controller_infos'] = controller_infos
        execute_info.append(step_info)
        
        # replan이 필요한지 확인 후 종료.
        curr_pos = movable_obs['position']
        if distance_transform_edt(1 - _affordance_map)[tuple(curr_pos)] <= 2:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached target; terminating {bcolors.ENDC}')
          break
    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished executing path via controller{bcolors.ENDC}')


    # 최종 타겟 위치로 ee 조정 (객체 이동이 아닐 경우)
    # make sure we are at the final target position and satisfy any additional parametrization
    # (skip if we are specifying object-centric motion)
    if not object_centric:
      try:
        # traj_world: world_xyz, rotation, velocity, gripper
        ee_pos_world = traj_world[-1][0]
        ee_rot_world = traj_world[-1][1]
        ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
        ee_speed = traj_world[-1][2]
        gripper_state = traj_world[-1][3]
      except:
        # evaluate latest voxel map
        _rotation_map = rotation_map()
        _velocity_map = velocity_map()
        _gripper_map = gripper_map()
        # get last ee pose
        ee_pos_world = self._env.get_ee_pos()
        ee_pos_voxel = self.get_ee_pos()
        ee_rot_world = _rotation_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
        ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
        ee_speed = _velocity_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
        gripper_state = _gripper_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
      # move to the final target
      self._env.apply_action(np.concatenate([ee_pose_world, [gripper_state]]))

    return execute_info
  
  # cm를 Voxel 인덱스로 변환.
  def cm2index(self, cm, direction): # direction 변환할 방향.
    # 방향이 x,y,z 문자열로 주어진 경우.
    if isinstance(direction, str) and direction == 'x':
      x_resolution = self._resolution[0] * 100  # resolution is in m, we need cm 해당도는 m단위였음. 
      return int(cm / x_resolution) # 인덱스화.
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = self._resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = self._resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # 방향이 특정 벡터 3d numpy 배열로 주어진 경우
      # calculate index along the direction
      assert isinstance(direction, np.ndarray) and direction.shape == (3,)
      direction = normalize_vector(direction) # 단위벡터화.
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = self.cm2index(x_cm, 'x')
      y_index = self.cm2index(y_cm, 'y')
      z_index = self.cm2index(z_cm, 'z')
      return np.array([x_index, y_index, z_index])
  
  # Voxel 인덱스를 cm로 변환
  def index2cm(self, index, direction=None):
    if direction is None:
      average_resolution = np.mean(self._resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = self._resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = self._resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = self._resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
  
  # 벡터를 쿼터니안으로 변환
  def pointat2quat(self, vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)

  # 특정 voxel 좌표 및 반경 내의 voxel 값을 설정.
  def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value # 특정 voxel좌표.
    if radius_cm > 0:
      radius_x = self.cm2index(radius_cm, 'x')
      radius_y = self.cm2index(radius_cm, 'y')
      radius_z = self.cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    return voxel_map # 그 주변부의 voxel 값 세팅.
  
  def get_empty_affordance_map(self):
    return self._get_default_voxel_map('target')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

  def get_empty_avoidance_map(self):
    return self._get_default_voxel_map('obstacle')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_rotation_map(self):
    return self._get_default_voxel_map('rotation')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_velocity_map(self):
    return self._get_default_voxel_map('velocity')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_gripper_map(self):
    return self._get_default_voxel_map('gripper')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  # 로봇을 기본 위치로 초기화.
  def reset_to_default_pose(self):
     self._env.reset_to_default_pose()
  
  # ======================================================
  # == helper functions
  # ======================================================
  def _world_to_voxel(self, world_xyz):
    _world_xyz = world_xyz.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    voxel_xyz = pc2voxel(_world_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return voxel_xyz

  def _voxel_to_world(self, voxel_xyz):
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    world_xyz = voxel2pc(voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return world_xyz

  def _points_to_voxel_map(self, points):
    """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
    _points = points.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

  def _get_voxel_center(self, voxel_map):
    """calculte the center of the voxel map where value is 1"""
    voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
    return voxel_center

  def _get_scene_collision_voxel_map(self):
    collision_points_world, _ = self._env.get_scene_3d_obs(ignore_robot=True)
    collision_voxel = self._points_to_voxel_map(collision_points_world)
    return collision_voxel

  def _get_default_voxel_map(self, type='target'):
    """returns default voxel map (defaults to current state)"""
    def fn_wrapper():
      if type == 'target':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'obstacle':  # for LLM to do customization
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'velocity':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size))
      elif type == 'gripper':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size)) * self._env.get_last_gripper_action()
      elif type == 'rotation':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size, 4))
        voxel_map[:, :, :] = self._env.get_ee_quat()
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper
  

  # 로봇의 이동경로path(Planner가 생성한 단순 voxel 좌표 리스트)를실제 실행가능 trajectory로 변환, 장애물 회피를 처리하는 역할.
  def _path2traj(self, path, rotation_map, velocity_map, gripper_map):
    """
    convert path (generated by planner) to trajectory (used by controller)
    path only contains a sequence of voxel coordinates, while trajectory parametrize the motion of the end-effector with rotation, velocity, and gripper on/off command
    """
    # convert path to trajectory
    traj = []
    for i in range(len(path)):
      # get the current voxel position
      voxel_xyz = path[i]
      # get the current world position
      world_xyz = self._voxel_to_world(voxel_xyz)
      voxel_xyz = np.round(voxel_xyz).astype(int)
      # get the current rotation (in world frame)
      rotation = rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current velocity
      velocity = velocity_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current on/off
      gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # LLM might specify a gripper value change, but sometimes EE may not be able to reach the exact voxel, so we overwrite the gripper value if it's close enough (TODO: better way to do this?)
      if (i == len(path) - 1) and not (np.all(gripper_map == 1) or np.all(gripper_map == 0)):
        # get indices of the less common values
        less_common_value = 1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
        less_common_indices = np.where(gripper_map == less_common_value)
        less_common_indices = np.array(less_common_indices).T
        # get closest distance from voxel_xyz to any of the indices that have less common value
        closest_distance = np.min(np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0))
        # if the closest distance is less than threshold, then set gripper to less common value
        if closest_distance <= 3:
          gripper = less_common_value
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] overwriting gripper to less common value for the last waypoint{bcolors.ENDC}')
      # add to trajectory
      traj.append((world_xyz, rotation, velocity, gripper))

    # append the last waypoint a few more times for the robot to stabilize
    for _ in range(2): # 마지막 웨이포인트 2개 추가 -> 안정스탑.
      traj.append((world_xyz, rotation, velocity, gripper))
    return traj # 실행 가능한 Trajectory 리스트 [(위치, 회전, 속도, 그리퍼 상태)]
  


  # 장애물(Avoidance) 맵을 전처리하여 목표 및 시작 지점 주변의 장애물 영향을 줄이는 함수.
  def _preprocess_avoidance_map(self, avoidance_map, affordance_map, movable_obs):
    # collision avoidance
    scene_collision_map = self._get_scene_collision_voxel_map() # 환경의 기존 장애물 맵
    # anywhere within 15/100 indices of the target is ignored (to guarantee that we can reach the target)
    # 목표 지점에서 15% (0.15 * map_size) 범위 내의 장애물 제거
    ignore_mask = distance_transform_edt(1 - affordance_map)
    scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    # anywhere within 15/100 indices of the start is ignored
    #  시작 지점(Start) 15%주변의 장애물 제거
    try:
      # 객체가 차지하는 공간(occupancy_map)에서 15% 범위 내 장애물 제거
      ignore_mask = distance_transform_edt(1 - movable_obs['occupancy_map'])
      scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    except KeyError:
      # 객체의 위치(Position)만 있는 경우 (occupancy_map 없는 경우)
      start_pos = movable_obs['position']
      ignore_mask = np.ones_like(avoidance_map)
      # 시작 위치에서 10% (0.1 * map_size) 범위 내 장애물 제거
      ignore_mask[start_pos[0] - int(0.1 * self._map_size):start_pos[0] + int(0.1 * self._map_size),
                  start_pos[1] - int(0.1 * self._map_size):start_pos[1] + int(0.1 * self._map_size),
                  start_pos[2] - int(0.1 * self._map_size):start_pos[2] + int(0.1 * self._map_size)] = 0
      scene_collision_map *= ignore_mask # 장애물 제거
    
    # 기존 장애물 맵을 업데이트하여 최종 장애물 맵 생성
    avoidance_map += scene_collision_map
    avoidance_map = np.clip(avoidance_map, 0, 1) # 값 범위 (0~1)로 제한
    return avoidance_map # 최종 장애물 맵 반환

def setup_LMP(env, general_config, debug=False):

  # LMP(Language Motion Planner) 설정을 위한 함수
  # env: 환경 객체 -> playground.ipynb에서 VoxPoserRLBench를 사용함
  # general_config: 전체 설정 정보를 담은 딕셔너리
  # debug: 디버그 모드 여부 (기본값: False)

  # >>>>설정 파일에서 각 컴포넌트별 설정 추출<<<
  # controller (컨트롤러)
  # - 로봇의 실제 동작을 제어하는 컴포넌트
  # - 주요기능
  # -- 로봇 관절의 위치/속도 제어
  # -- 움직임의 부드러움 조절
  # -- 안전 제한 설정
  # -- 피드백 제어 구현 (그러면 여기서 피드백이란? 피드백 제어란?)
  # planner (플래너)
  # - 로봇의 동작을 계획하는 컴포넌트
  # - 주요 기능
  # -- 경로 계획
  # -- 충돌 회피
  # -- 작업 순서 최적화
  # -- 동작 퀘적 생성
  # lmp_env (LMP 환경)
  # - Learned Motion Primitives 환경 설정
  # - 주요 기능
  # -- 학습된 동작의 실행 환경 정의
  # -- 상태 공간 설정
  # -- 보상 함수 정의
  # -- 관찰 공간 설정
  # lmps (LMP 시스템)
  # - Learned Motion Primitivs 전체 시스템 설정
  # - 주요 기능
  # -- 동작 학습 파라미터
  # -- 모델 구조 설정
  # -- 학습 알고리즘 파라미터
  # -- 동작 생성 설정
  # env_name (환경 이름)
  # - 사용할 시뮬레이션 환경 지정
  # - 주요 기능
  # -- 특정 작업 환경 로드
  # -- 환경별 특수 설정 적용
  # -- 시뮬레이션 파라미터 설정
  controller_config = general_config['controller']  # 컨트롤러 설정
  planner_config = general_config['planner']       # 플래너 설정
  lmp_env_config = general_config['lmp_config']['env']  # LMP 환경 설정
  lmps_config = general_config['lmp_config']['lmps']    # LMP 시스템 설정
  env_name = general_config['env_name']                 # 환경 이름

  # LMP 환경 래퍼 생성
  # LMP 환경 래퍼
  # - 기본 환경(env)을 LMP 시스템에서 사용할 수 있도록 감싸는 인터페이스
  # - 여러 설정들(환경, 컨트롤러, 플래너)을 통합하여 관리
  # 귱금한 점 - lmps_config는 왜 안 넣지?
  lmp_env = LMP_interface(env, lmp_env_config, controller_config, planner_config, env_name=env_name)

  # LMP가 사용할 수 있는 고정 API 설정
  fixed_vars = {
      'np': np,  # numpy
      'euler2quat': transforms3d.euler.euler2quat,  # 오일러각->쿼터니언 변환
      'quat2euler': transforms3d.euler.quat2euler,  # 쿼터니언->오일러각 변환  
      'qinverse': transforms3d.quaternions.qinverse, # 쿼터니언 역변환
      'qmult': transforms3d.quaternions.qmult,      # 쿼터니언 곱셈
  }

  # LMP가 사용할 수 있는 환경 API 설정 
  variable_vars = {
      k: getattr(lmp_env, k)
      for k in dir(lmp_env) if callable(getattr(lmp_env, k)) and not k.startswith("_")
  }

  # 하위 레벨 LMP 컴포넌트들 생성
  # ['lmp_config']['lmps'] -> planner, composer, 
  lmp_names = [name for name in lmps_config.keys() if not name in ['composer', 'planner', 'config']]
  low_level_lmps = {
      k: LMP(k, lmps_config[k], fixed_vars, variable_vars, debug, env_name)
      for k in lmp_names
  }
  variable_vars.update(low_level_lmps)  # 하위 레벨 LMP들을 variable_vars에 추가

  # 스킬 레벨 구성을 위한 composer LMP 생성
  composer = LMP(
      'composer', lmps_config['composer'], fixed_vars, variable_vars, debug, env_name
  )
  variable_vars['composer'] = composer  # composer를 variable_vars에 추가

  # 고수준 언어 명령을 처리하는 task planner LMP 생성
  task_planner = LMP(
      'planner', lmps_config['planner'], fixed_vars, variable_vars, debug, env_name
  )

  # 모든 LMP 컴포넌트들을 하나의 딕셔너리로 구성
  lmps = {
      'plan_ui': task_planner,
      'composer_ui': composer,
  }
  lmps.update(low_level_lmps)  # 하위 레벨 LMP들 추가

  # LMP 시스템과 환경 래퍼 반환
  return lmps, lmp_env


# ======================================================
# jit-ready functions (for faster replanning time, need to install numba and add "@njit")
# ======================================================
def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """voxelize a point cloud"""
  pc = pc.astype(np.float32)
  # make sure the point is within the voxel bounds
  pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxels = (pc - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxels)
  voxels = np.round(voxels, 0, _out).astype(np.int32)
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  return voxels

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """given point cloud, create a fixed size voxel map, and fill in the voxels"""
  points = points.astype(np.float32)
  voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
  voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
  # make sure the point is within the voxel bounds
  points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max) # points내의 요소들에 대해, min값보다 작은 건 min으로 바꾸고, max보다 크면 max로 바꿈.
  # voxelize
  voxel_xyz = (points - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1) # 
  # to integer
  _out = np.empty_like(voxel_xyz)
  points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32) # 복셀화된 값, 반올림.
  voxel_map = np.zeros((map_size, map_size, map_size))
  for i in range(points_vox.shape[0]):
      voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
  return voxel_map