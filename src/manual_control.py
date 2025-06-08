#!/usr/bin/env python
# 参考：https://github.com/marcgpuig/carla_py_clients/blob/master/imu_plot.py
# 版权信息：2019年巴塞罗那自治大学计算机视觉中心（CVC），基于MIT协议发布

# 该脚本实现了通过键盘手动控制CARLA仿真中的车辆，包含传感器管理、物理控制、HUD显示等功能
# 若需手动操控，需在命令行运行：python manual_control.py（需位于指定目录）

"""
键盘控制说明：
- 方向键/WASD：控制油门、刹车、转向
- Q：切换倒车模式
- 空格：手刹
- P：切换自动驾驶
- M：切换手动/自动变速箱
- ,/.：升降挡（手动模式）
- L：切换灯光类型，Shift+L：远光灯，Z/X：转向灯
- TAB：切换传感器位置，`/N：切换传感器，1-9：直接选择传感器
- G：切换雷达可视化
- C：切换天气（Shift+C反向）
- Backspace：切换车辆
- O：开关所有车门
- T：切换车辆遥测显示
- V：切换地图层（Shift+V反向），B：加载/卸载当前层
- R：切换图像录制，Ctrl+R：录制/回放仿真
- F1：切换HUD，H/?：显示帮助，ESC：退出
"""

from __future__ import print_function  # 导入未来特性，支持Python3的print函数风格

# ==============================================================================
# -- 查找CARLA模块路径 ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys  # 导入系统模块，用于路径操作和进程管理

# 获取当前进程ID（用于调试追踪）
process_id = os.getpid()
print("当前进程ID：", process_id)

try:
    # 将CARLA的Python API库路径添加到系统搜索路径
    # 根据Python版本和操作系统架构匹配对应的egg文件
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass  # 未找到匹配文件时忽略错误

# ==============================================================================
# -- 依赖库导入 ----------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc  # 导入颜色转换器，用于传感器数据处理

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref  # 弱引用模块，避免循环引用导致内存泄漏

try:
    import pygame  # 导入游戏开发库，用于创建窗口和处理输入
    from pygame.locals import *  # 导入PyGame常量
except ImportError:
    raise RuntimeError('无法导入pygame，请确保已安装pygame库')

try:
    import numpy as np  # 导入数值计算库，用于传感器数据处理
except ImportError:
    raise RuntimeError('无法导入numpy，请确保已安装numpy库')


# ==============================================================================
# -- 全局工具函数 --------------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    """获取所有天气预设，将类名转换为易读名称"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')  # 正则表达式拆分PascalCase字符串
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))  # 格式化名称
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]  # 提取天气参数名
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]  # 返回（天气对象，名称）元组列表

def get_actor_display_name(actor, truncate=250):
    """将Actor类型ID（如vehicle.tesla.model3）转换为易读名称（如Tesla Model3）"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])  # 格式化类型名
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name  # 截断长名称

def get_actor_blueprints(world, filter, generation):
    """根据过滤条件和代数获取Actor蓝图列表"""
    bps = world.get_blueprint_library().filter(filter)  # 按类型过滤蓝图

    if generation.lower() == "all":
        return bps  # 返回所有符合条件的蓝图

    if len(bps) == 1:
        return bps  # 若仅一个蓝图，忽略代数限制

    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            return [x for x in bps if int(x.get_attribute('generation')) == int_generation]  # 按代数筛选
        else:
            print("警告！无效的Actor代数，无法生成Actor")
            return []
    except:
        print("警告！无效的Actor代数，无法生成Actor")
        return []


# ==============================================================================
# -- 世界管理类 ----------------------------------------------------------------
# ==============================================================================

class World(object):
    """CARLA世界管理类，负责车辆生成、传感器管理、天气控制等"""
    def __init__(self, carla_world, hud, args):
        self.world = carla_world  # CARLA世界对象
        self.sync = args.sync      # 是否启用同步模式
        self.actor_role_name = args.rolename  # 车辆角色名（用于标识）

        try:
            self.map = self.world.get_map()  # 获取当前地图
        except RuntimeError as e:
            print(f'运行时错误：{e}')
            print('  服务器无法发送OpenDRIVE文件，请检查文件是否存在且正确')
            sys.exit(1)

        self.hud = hud  # HUD显示对象
        self.player = None  # 玩家控制的车辆

        # 传感器对象
        self.collision_sensor = None    # 碰撞传感器
        self.lane_invasion_sensor = None # 车道入侵传感器
        self.gnss_sensor = None         # GNSS传感器
        self.imu_sensor = None          # IMU传感器
        self.radar_sensor = None        # 雷达传感器
        self.camera_manager = None      # 相机管理器

        self._weather_presets = find_weather_presets()  # 获取天气预设列表
        self._weather_index = 0  # 当前天气索引
        self._actor_filter = args.filter  # 车辆类型过滤条件
        self._actor_generation = args.generation  # 车辆代数
        self._gamma = args.gamma  # 相机伽马校正值

        self.restart()  # 初始化车辆和传感器
        self.world.on_tick(hud.on_world_tick)  # 绑定世界更新回调

        # 录制相关
        self.recording_enabled = False
        self.recording_start = 0

        # 控制状态
        self.constant_velocity_enabled = False  # 定速巡航状态
        self.show_vehicle_telemetry = False     # 车辆遥测显示状态
        self.doors_are_open = False             # 车门状态
        self.current_map_layer = 0              # 当前地图层索引
        self.map_layer_names = [
            carla.MapLayer.NONE, carla.MapLayer.Buildings, carla.MapLayer.Decals,
            carla.MapLayer.Foliage, carla.MapLayer.Ground, carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles, carla.MapLayer.Props, carla.MapLayer.StreetLights,
            carla.MapLayer.Walls, carla.MapLayer.All
        ]  # 地图层名称列表

    def restart(self):
        """重置车辆和传感器，重新生成车辆"""
        self.player_max_speed = 1.589         # 车辆最大速度（普通模式）
        self.player_max_speed_fast = 3.713    # 车辆最大速度（高速模式）

        # 保留相机配置（若存在）
        cam_index = self.camera_manager.index if self.camera_manager else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager else 0

        # 获取符合条件的车辆蓝图
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("未找到符合条件的车辆蓝图")

        blueprint = random.choice(blueprint_list)  # 随机选择一个蓝图
        blueprint.set_attribute('role_name', self.actor_role_name)  # 设置角色名

        # 配置蓝图属性（颜色、是否无敌等）
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # 生成车辆
        if self.player:
            # 在原位置上方生成新车辆
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation = carla.Rotation(roll=0, pitch=0)
            self.destroy()  # 销毁旧车辆和传感器
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # 处理生成失败的情况（循环直到成功）
        while not self.player:
            if not self.map.get_spawn_points():
                print("地图中没有可用生成点，请在UE4场景中添加车辆生成点")
                sys.exit(1)
            spawn_point = random.choice(self.map.get_spawn_points())
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # 创建传感器
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

        # 显示车辆类型通知
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """切换天气预设（reverse为True时反向切换）"""
        self._weather_index = (self._weather_index - 1) if reverse else (self._weather_index + 1)
        self._weather_index %= len(self._weather_presets)  # 循环索引
        preset = self._weather_presets[self._weather_index]
        self.hud.notification(f'天气：{preset[1]}')  # HUD显示天气名称
        self.world.set_weather(preset[0])  # 应用天气设置

    def next_map_layer(self, reverse=False):
        """切换地图层（reverse为True时反向切换）"""
        self.current_map_layer = (self.current_map_layer - 1) if reverse else (self.current_map_layer + 1)
        self.current_map_layer %= len(self.map_layer_names)  # 循环索引
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification(f'选中地图层：{selected}')  # HUD显示地图层名称

    def load_map_layer(self, unload=False):
        """加载或卸载当前地图层（unload为True时卸载）"""
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.world.unload_map_layer(selected)  # 卸载图层
            self.hud.notification(f'卸载地图层：{selected}')
        else:
            self.world.load_map_layer(selected)  # 加载图层
            self.hud.notification(f'加载地图层：{selected}')

    def toggle_radar(self):
        """切换雷达传感器（创建或销毁）"""
        if not self.radar_sensor:
            self.radar_sensor = RadarSensor(self.player)  # 创建雷达传感器
        elif self.radar_sensor.sensor:
            self.radar_sensor.sensor.destroy()  # 销毁雷达传感器
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        """修改车辆物理属性（启用轮扫碰撞检测）"""
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True  # 启用轮扫碰撞
            actor.apply_physics_control(physics_control)
        except:
            pass  # 非车辆对象时忽略

    def tick(self, clock):
        """更新HUD显示"""
        self.hud.tick(self, clock)

    def render(self, display):
        """渲染相机画面和HUD"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """销毁所有传感器"""
        if self.camera_manager and self.camera_manager.sensor:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None

    def destroy(self):
        """销毁所有实体（车辆和传感器）"""
        if self.radar_sensor:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor
        ]
        for sensor in sensors:
            if sensor:
                sensor.stop()
                sensor.destroy()
        if self.player:
            self.player.destroy()


# ==============================================================================
# -- 键盘控制类 --------------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """键盘输入处理类，负责解析按键并生成控制指令"""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot  # 自动驾驶状态
        self._ackermann_enabled = False  # 阿克曼转向模式状态
        self._ackermann_reverse = 1  # 阿克曼倒车方向

        # 根据角色类型初始化控制对象（车辆或行人）
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()  # 车辆控制对象
            self._ackermann_control = carla.VehicleAckermannControl()  # 阿克曼控制对象
            self._lights = carla.VehicleLightState.NONE  # 灯光状态
            world.player.set_autopilot(self._autopilot_enabled)  # 初始化自动驾驶状态
            world.player.set_light_state(self._lights)  # 初始化灯光状态
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()  # 行人控制对象
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("不支持的角色类型")

        self._steer_cache = 0.0  # 转向缓存（用于平滑转向）
        world.hud.notification("按'H'或'?'获取帮助", seconds=4.0)  # 显示帮助提示

    def parse_events(self, client, world, clock, sync_mode):
        """解析PyGame事件（键盘输入）"""
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights

        for event in pygame.event.get():
            if event.type == QUIT:
                return True  # 关闭窗口时退出
            elif event.type == KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True  # ESC或Ctrl+Q退出

                # 处理车辆重置、天气切换、传感器切换等功能键
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                # ...（其他功能键处理，代码中已省略部分以保持简洁）

        if not self._autopilot_enabled:
            # 手动控制模式下处理方向和速度
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0  # 根据档位设置倒车状态
                # 更新灯光状态（刹车灯、倒车灯等）
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights:
                    world.player.set_light_state(current_lights)
                    self._lights = current_lights
                # 应用控制指令（普通模式或阿克曼模式）
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    self._control = world.player.get_control()  # 更新控制状态
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

        return False

    def _parse_vehicle_keys(self, keys, milliseconds):
        """解析车辆控制按键（油门、刹车、转向等）"""
        # 油门/刹车控制
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.0)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        # 转向控制（平滑处理）
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache = max(self._steer_cache - steer_increment, -0.7)
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache = min(self._steer_cache + steer_increment, 0.7)
        else:
            self._steer_cache = 0.0

        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]  # 手刹
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    @staticmethod
    def _is_quit_shortcut(key):
        """判断是否为退出快捷键（ESC或Ctrl+Q）"""
        return key == K_ESCAPE or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD显示类 ----------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """抬头显示类，负责绘制实时信息"""
    def __init__(self, width, height):
        self.dim = (width, height)  # 窗口尺寸
        self._font_mono = pygame.font.Font(pygame.font.match_font('mono'), 14)  # 等宽字体
        self._notifications = FadingText(pygame.font.Font(pygame.font.get_default_font(), 20), (width, 40), (0, height-40))  # 通知文本
        self.help = HelpText(self._font_mono, width, height)  # 帮助文本

        # 性能指标
        self.server_fps = 0       # 服务器帧率
        self.frame = 0            # 当前帧数
        self.simulation_time = 0  # 仿真时间

        self._show_info = True    # 是否显示信息面板
        self._info_text = []      # 信息内容列表
        self._server_clock = pygame.time.Clock()  # 服务器时钟

    def on_world_tick(self, timestamp):
        """世界更新时回调，更新性能指标"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """更新HUD显示内容"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return

        # 获取车辆状态
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = self._get_compass_heading(compass)  # 获取罗盘方向

        # 构建信息文本
        self._info_text = [
            f'服务器: {self.server_fps:16.0f} FPS',
            f'客户端: {clock.get_fps():16.0f} FPS',
            '',
            f'车辆: {get_actor_display_name(world.player, 20):20s}',
            f'地图: {world.map.name.split("/")[-1]:20s}',
            f'仿真时间: {datetime.timedelta(seconds=int(self.simulation_time)):12s}',
            '',
            f'速度: {3.6*math.hypot(v.x, v.y, v.z):15.0f} km/h',
            f'罗盘: {compass:17.0f}\N{DEGREE SIGN} {heading:2s}',
            f'加速度: {world.imu_sensor.accelerometer}',
            f'陀螺仪: {world.imu_sensor.gyroscope}',
            f'位置: ({t.location.x:5.1f}, {t.location.y:5.1f})',
            f'GNSS: ({world.gnss_sensor.lat:2.6f}, {world.gnss_sensor.lon:3.6f})',
            f'高度: {t.location.z:18.0f} m',
            ''
        ]
        # 添加车辆控制状态（如油门、档位等）
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('油门:', c.throttle, 0.0, 1.0),
                ('转向:', c.steer, -1.0, 1.0),
                ('刹车:', c.brake, 0.0, 1.0),
                ('倒车:', c.reverse),
                ('手刹:', c.hand_brake),
                ('手动模式:', c.manual_gear_shift),
                f'档位: {"R" if c.gear < 0 else "N" if c.gear == 0 else c.gear}'
            ]
        # ...（其他信息添加，代码中已省略部分以保持简洁）

    def _get_compass_heading(self, angle):
        """根据罗盘角度返回方向缩写（如N、SE等）"""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((angle + 22.5) % 360 / 45)
        return directions[index]

    # ...（渲染相关方法，代码中已省略部分以保持简洁）


# ==============================================================================
# -- 传感器类（碰撞、车道入侵、GNSS、IMU、雷达）--------------------------------
# ==============================================================================

class CollisionSensor(object):
    """碰撞传感器，记录碰撞历史并在HUD显示"""
    def __init__(self, parent_actor, hud):
t_actor, hud, gamma_correction):

        self.sensor = None
        self.history = []  # 碰撞历史（帧号，强度）
        self._parent = parent_actor
        self.hud = hud

        # 创建传感器并绑定回调
        bp = parent_actor.get_world().get_blueprint_library().find('sensor.other.collision')
        self.sensor = parent_actor.get_world().spawn_actor(bp, carla.Transform(), attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda e: CollisionSensor._on_collision(weak_self, e))

    @staticmethod
    def _on_collision(weak_self, event):
        """碰撞事件回调，记录碰撞强度并显示通知"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f'碰撞：{actor_type}')
        intensity = math.hypot(*event.normal_impulse)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)  # 限制历史记录长度


# ==============================================================================
# -- 主循环和程序入口 ----------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """主循环函数，负责初始化、事件处理和渲染"""
    pygame.init()
    display = pygame.display.set_mode((args.width, args.height), HWSURFACE | DOUBLEBUF)
    hud = HUD(args.width, args.height)
    world = World(client.get_world(), hud, args)
    controller = KeyboardControl(world, args.autopilot)

    while True:
        if args.sync:
            world.world.tick()  # 同步模式下手动推进仿真
        controller.parse_events(client, world, clock, args.sync)
        world.tick(clock)
        world.render(display)
        pygame.display.flip()
        clock.tick_busy_loop(60)

if __name__ == '__main__':
    """程序入口，解析命令行参数并启动主循环"""
    argparser = argparse.ArgumentParser(description='CARLA手动控制客户端')
    argparser.add_argument('--host', default='127.0.0.1', help='服务器IP')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='服务器端口')
    argparser.add_argument('--res', default='1280x720', help='窗口分辨率')
    # ...（其他参数解析，代码中已省略部分以保持简洁）
    args = argparser.parse_args()
    args.width, args.height = map(int, args.res.split('x'))
    main()
