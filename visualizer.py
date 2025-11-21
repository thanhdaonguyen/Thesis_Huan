import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from typing import Optional, Dict, List, Tuple
import argparse

from environment import CommEnvironment
from maddpg import MADDPGAgent


class Camera3D:
    """3D Camera controller with smooth movements"""
    def __init__(self):
        self.distance = 1200
        self.theta = math.pi / 4  # Horizontal rotation
        self.phi = math.pi / 6    # Vertical rotation
        self.target = np.array([250.0, 0.0, 250.0])  # Center of 500x500 area
        self.position = self.calculate_position()
        
    def calculate_position(self):
        x = self.target[0] + self.distance * math.cos(self.phi) * math.cos(self.theta)
        y = self.target[1] + self.distance * math.sin(self.phi)
        z = self.target[2] + self.distance * math.cos(self.phi) * math.sin(self.theta)
        return np.array([x, y, z])
    
    def rotate(self, delta_theta, delta_phi):
        self.theta += delta_theta
        self.phi = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.phi + delta_phi))
        self.position = self.calculate_position()
    
    def zoom(self, delta):
        self.distance = max(200, min(3000, self.distance + delta))
        self.position = self.calculate_position()
    
    def pan(self, dx, dy):
        right = np.array([math.sin(self.theta), 0, -math.cos(self.theta)])
        up = np.array([0, 1, 0])
        self.target += right * dx + up * dy
        self.position = self.calculate_position()
    
    def apply(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            self.target[0], self.target[1], self.target[2],
            0, 1, 0
        )


class AntennaModel:
    """Procedural antenna model for SU/DU"""
    @staticmethod
    def draw_transmitter(height=15, is_active=True, amplitude=0.5):
        """Draw a transmitter antenna (SU)"""
        # Rotate to point upward (Y-axis)
        glRotatef(-90, 1, 0, 0)
        
        # Base
        glColor3f(0.3, 0.3, 0.35)
        quad = gluNewQuadric()
        gluCylinder(quad, 3, 3, 2, 12, 1)
        
        # Mast
        glTranslatef(0, 0, 2)
        glColor3f(0.5, 0.5, 0.55)
        gluCylinder(quad, 1, 1, height, 8, 1)
        
        # Antenna dish/array at top
        glTranslatef(0, 0, height)
        if is_active:
            # Color based on amplitude
            intensity = 0.3 + amplitude * 0.7
            glColor3f(0.2, intensity, 0.8)
        else:
            glColor3f(0.3, 0.3, 0.4)
        
        glRotatef(90, 1, 0, 0)
        gluCylinder(quad, 4, 2, 3, 16, 1)
        gluDeleteQuadric(quad)
    
    @staticmethod
    def draw_receiver(height=12):
        """Draw a receiver antenna (DU)"""
        # Rotate to point upward (Y-axis)
        glRotatef(-90, 1, 0, 0)
        
        # Base platform
        glColor3f(0.2, 0.3, 0.2)
        quad = gluNewQuadric()
        gluCylinder(quad, 5, 5, 1, 12, 1)
        
        # Mast
        glTranslatef(0, 0, 1)
        glColor3f(0.4, 0.5, 0.4)
        gluCylinder(quad, 0.8, 0.8, height, 8, 1)
        
        # Receiver array
        glTranslatef(0, 0, height)
        glColor3f(0.3, 0.6, 0.4)
        gluSphere(quad, 3, 12, 12)
        gluDeleteQuadric(quad)
    
    @staticmethod
    def draw_jammer(time, power_scale=1.0):
        """Draw the jammer with menacing appearance"""
        # Rotate to point upward (Y-axis)
        glRotatef(-90, 1, 0, 0)
        
        quad = gluNewQuadric()
        
        # Pulsing effect
        pulse = 0.8 + 0.2 * math.sin(time * 3)
        
        # Base
        glColor3f(0.4, 0.1, 0.1)
        gluCylinder(quad, 8, 8, 3, 16, 1)
        
        # Main body
        glTranslatef(0, 0, 3)
        glColor3f(0.6, 0.15, 0.15)
        gluCylinder(quad, 6, 6, 20, 16, 1)
        
        # Top antenna array
        glTranslatef(0, 0, 20)
        glColor3f(pulse, 0.1, 0.1)
        gluSphere(quad, 8 * power_scale, 16, 16)
        
        # Danger indicators (rotating)
        for i in range(4):
            glPushMatrix()
            glRotatef(i * 90 + time * 50, 0, 0, 1)
            glTranslatef(10, 0, 0)
            glColor3f(1.0, 0.2, 0.2)
            gluSphere(quad, 2, 8, 8)
            glPopMatrix()
        
        gluDeleteQuadric(quad)
    
    @staticmethod
    def draw_relay_station():
        """Draw the relay base station (rBS)"""
        # Rotate to point upward (Y-axis)
        glRotatef(-90, 1, 0, 0)
        
        quad = gluNewQuadric()
        
        # Base platform
        glColor3f(0.3, 0.4, 0.5)
        gluCylinder(quad, 12, 12, 2, 20, 1)
        
        # Central tower
        glTranslatef(0, 0, 2)
        glColor3f(0.5, 0.6, 0.7)
        gluCylinder(quad, 4, 3, 30, 16, 1)
        
        # Top platform
        glTranslatef(0, 0, 30)
        glColor3f(0.4, 0.5, 0.6)
        gluCylinder(quad, 8, 8, 2, 16, 1)
        
        # Relay antennas (4 directions)
        glTranslatef(0, 0, 2)
        for i in range(4):
            glPushMatrix()
            glRotatef(i * 90, 0, 0, 1)
            glTranslatef(8, 0, 2)
            glColor3f(0.3, 0.7, 0.9)
            glRotatef(90, 0, 1, 0)
            gluCylinder(quad, 1.5, 1.5, 6, 12, 1)
            glTranslatef(0, 0, 6)
            gluSphere(quad, 2, 12, 12)
            glPopMatrix()
        
        gluDeleteQuadric(quad)


class CommVisualizer3D:
    """3D Visualizer for Anti-Jamming Communication Environment"""
    
    def __init__(self, env: CommEnvironment, agent: Optional[MADDPGAgent] = None,
                 window_width: int = 1600, window_height: int = 900,
                 render_fps: int = 60):
        pygame.init()
        
        self.env = env
        self.agent = agent
        self.window_width = window_width
        self.window_height = window_height
        self.render_fps = render_fps
        
        # OpenGL setup
        self.display = pygame.display.set_mode(
            (window_width, window_height), 
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Anti-Jamming Communication 3D Visualizer")
        
        # Camera
        self.camera = Camera3D()
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glClearColor(0.05, 0.05, 0.1, 1.0)  # Dark background
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [500, 400, 500, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.5, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, window_width / window_height, 1, 5000)
        
        # View options
        self.show_connections = True
        self.show_signal_cones = True
        self.show_sinr_map = True
        self.show_paths = True
        self.show_interference = True
        self.selected_agent = None
        
        # Mouse state
        self.mouse_dragging = False
        self.mouse_button = None
        self.last_mouse_pos = (0, 0)
        
        # Animation
        self.animation_time = 0.0
        self.clock = pygame.time.Clock()
        self.paused = False
        self.step_mode = False
        
        # Agent paths (trails)
        self.su_paths = {i: [] for i in range(env.num_agents)}
        self.du_paths = {i: [] for i in range(env.num_agents)}
        self.max_path_length = 150
        
        # Fonts
        self.font_small = pygame.font.SysFont("Consolas", 12)
        self.font_medium = pygame.font.SysFont("Consolas", 14)
        self.font_large = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # Performance tracking
        self.episode_count = 0
        self.step_count = 0
        self.last_info = {}
        
    def draw_ground(self):
        """Draw ground plane with grid"""
        glDisable(GL_LIGHTING)
        
        # Main ground plane
        glColor4f(0.08, 0.12, 0.15, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(0, -2, 0)
        glVertex3f(500, -2, 0)
        glVertex3f(500, -2, 500)
        glVertex3f(0, -2, 500)
        glEnd()
        
        # Grid lines
        glLineWidth(1)
        glColor4f(0.15, 0.2, 0.25, 0.6)
        glBegin(GL_LINES)
        for i in range(0, 501, 50):
            # Vertical lines
            glVertex3f(i, -1, 0)
            glVertex3f(i, -1, 500)
            # Horizontal lines
            glVertex3f(0, -1, i)
            glVertex3f(500, -1, i)
        glEnd()
        
        # Coordinate axes
        glLineWidth(3)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(0.8, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(100, 0, 0)
        # Y axis (green)
        glColor3f(0.2, 0.8, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 100, 0)
        # Z axis (blue)
        glColor3f(0.2, 0.2, 0.8)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 100)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_signal_cone(self, pos, target_pos, sinr, mode='d2d', is_active=True):
        """
        Draw signal cone from transmitter to receiver
        pos and target_pos are (x, y_height, z) tuples
        """
        if not self.show_signal_cones or not is_active:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        
        # Color based on SINR quality
        if sinr > 1.0:  # Good SINR
            color = (0.2, 0.8, 0.3, 0.3)
        elif sinr > 0.1:  # Acceptable
            color = (0.8, 0.8, 0.2, 0.3)
        else:  # Poor
            color = (0.8, 0.2, 0.2, 0.3)
        
        # Different style for relay mode
        if mode == 'relay':
            color = (color[0] * 0.7, color[1] * 0.7, color[2] + 0.3, 0.25)
        
        # Draw cone
        num_segments = 16
        cone_radius = 8
        
        glColor4f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(pos[0], pos[1], pos[2])  # Apex at transmitter (x, y_height, z)
        
        for i in range(num_segments + 1):
            angle = (i / num_segments) * 2 * np.pi
            x = target_pos[0] + cone_radius * np.cos(angle)
            z = target_pos[2] + cone_radius * np.sin(angle)
            glVertex3f(x, target_pos[1], z)  # Base circle at receiver
        glEnd()
        
        # Draw cone outline
        glLineWidth(2)
        glColor4f(color[0], color[1], color[2], 0.7)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            angle = (i / num_segments) * 2 * np.pi
            x = target_pos[0] + cone_radius * np.cos(angle)
            z = target_pos[2] + cone_radius * np.sin(angle)
            glVertex3f(x, target_pos[1], z)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_connection_line(self, pos1, pos2, sinr, mode='d2d', is_active=True):
        """
        Draw connection line between transmitter and receiver
        pos1 and pos2 are (x, y_height, z) tuples
        """
        if not self.show_connections or not is_active:
            return
        
        glDisable(GL_LIGHTING)
        
        # Line color based on SINR
        if sinr > 1.0:
            color = (0.2, 1.0, 0.3, 0.8)
        elif sinr > 0.1:
            color = (1.0, 1.0, 0.2, 0.8)
        else:
            color = (1.0, 0.2, 0.2, 0.8)
        
        # Dashed line for relay mode
        if mode == 'relay':
            glLineStipple(2, 0xAAAA)
            glEnable(GL_LINE_STIPPLE)
        
        glLineWidth(3)
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(pos1[0], pos1[1], pos1[2])  # (x, y_height, z)
        glVertex3f(pos2[0], pos2[1], pos2[2])  # (x, y_height, z)
        glEnd()
        
        if mode == 'relay':
            glDisable(GL_LINE_STIPPLE)
        
        glEnable(GL_LIGHTING)
    
    def draw_interference_indicator(self, pos, interference_level):
        """
        Draw interference indicator (pulsing red sphere)
        pos is (x, z, y_height) tuple - note different order for this function
        """
        if not self.show_interference or interference_level < 0.01:
            return
        
        glPushMatrix()
        glTranslatef(pos[0], pos[2] + 5, pos[1])  # (x, y_height+5, z)
        
        # Pulsing effect
        pulse = 0.7 + 0.3 * math.sin(self.animation_time * 5)
        intensity = min(1.0, interference_level * pulse)
        
        glColor4f(1.0, 0.2, 0.2, intensity * 0.6)
        quad = gluNewQuadric()
        gluSphere(quad, 3 + interference_level * 5, 12, 12)
        gluDeleteQuadric(quad)
        
        glPopMatrix()
    
    def draw_agents(self):
        """Draw all SUs (Secondary Users)"""
        for i in range(self.env.num_agents):
            su_pos = self.env.sus[i]
            
            glPushMatrix()
            glTranslatef(su_pos[0], 0, su_pos[1])
            
            # Determine if transmitting
            is_active = False
            amplitude = 0.0
            if self.last_info and 'amplitudes' in self.last_info:
                amplitude = self.last_info['amplitudes'][i]
                is_active = amplitude >= self.env.idle_threshold
            
            # Highlight selected agent
            if i == self.selected_agent:
                glColor4f(1.0, 1.0, 0.3, 0.5)
                quad = gluNewQuadric()
                gluCylinder(quad, 12, 12, 1, 20, 1)
                gluDeleteQuadric(quad)
            
            AntennaModel.draw_transmitter(height=15, is_active=is_active, amplitude=amplitude)
            glPopMatrix()
            
            # Draw ground connection line
            glDisable(GL_LIGHTING)
            glLineWidth(1)
            glColor4f(0.3, 0.5, 0.7, 0.3)
            glBegin(GL_LINES)
            glVertex3f(su_pos[0], 0, su_pos[1])
            glVertex3f(su_pos[0], 17, su_pos[1])  # Height of antenna
            glEnd()
            glEnable(GL_LIGHTING)
    
    def draw_destinations(self):
        """Draw all DUs (Destination Users)"""
        for i in range(self.env.num_agents):
            du_pos = self.env.dus[i]
            
            glPushMatrix()
            glTranslatef(du_pos[0], 0, du_pos[1])
            AntennaModel.draw_receiver(height=12)
            glPopMatrix()
    
    def draw_jammer(self):
        """Draw the jammer"""
        jammer_pos = self.env.jammer[0]
        
        glPushMatrix()
        glTranslatef(jammer_pos[0], 0, jammer_pos[1])
        
        # Jammer power visualization
        power_scale = self.env.jammer_power / 1.0  # Normalize
        AntennaModel.draw_jammer(self.animation_time, power_scale)
        
        glPopMatrix()
        
        # Draw jammer emission waves (pulsing circles at height)
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        
        wave_height = 23  # Top of jammer
        
        for r in range(3):
            radius = 50 + (r * 80) + (self.animation_time * 30) % 80
            alpha = 0.4 - (radius / 300)
            
            if alpha > 0:
                glColor4f(1.0, 0.2, 0.2, alpha)
                glBegin(GL_LINE_LOOP)
                for angle in range(0, 361, 10):
                    rad = math.radians(angle)
                    x = jammer_pos[0] + radius * math.cos(rad)
                    z = jammer_pos[1] + radius * math.sin(rad)
                    glVertex3f(x, wave_height, z)  # (x, y_height, z)
                glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_relay_station(self):
        """Draw the relay base station"""
        rbs_pos = self.env.rbs
        
        glPushMatrix()
        glTranslatef(rbs_pos[0], 0, rbs_pos[1])
        AntennaModel.draw_relay_station()
        glPopMatrix()
    
    def draw_paths(self):
        """Draw movement trails for agents"""
        if not self.show_paths:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        
        # SU paths (blue)
        for i, path in self.su_paths.items():
            if len(path) < 2:
                continue
            
            glBegin(GL_LINE_STRIP)
            for idx, pos in enumerate(path):
                alpha = (idx + 1) / len(path) * 0.6
                if i == self.selected_agent:
                    glColor4f(1.0, 1.0, 0.3, alpha)
                else:
                    glColor4f(0.3, 0.6, 1.0, alpha)
                glVertex3f(pos[0], 2, pos[1])
            glEnd()
        
        # DU paths (green)
        glLineWidth(1)
        for path in self.du_paths.values():
            if len(path) < 2:
                continue
            
            glBegin(GL_LINE_STRIP)
            for idx, pos in enumerate(path):
                alpha = (idx + 1) / len(path) * 0.4
                glColor4f(0.3, 1.0, 0.6, alpha)
                glVertex3f(pos[0], 2, pos[1])
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_connections_and_signals(self):
        """Draw all connections and signal visualizations"""
        if not self.last_info:
            return
        
        sinr_values = self.last_info.get('sinr_values', [])
        modes = self.last_info.get('modes', [])
        amplitudes = self.last_info.get('amplitudes', [])
        
        for i in range(self.env.num_agents):
            su_pos = self.env.sus[i]
            du_pos = self.env.dus[i]
            
            if i >= len(sinr_values):
                continue
            
            sinr = sinr_values[i]
            mode = modes[i] if i < len(modes) else 'd2d'
            amplitude = amplitudes[i] if i < len(amplitudes) else 0.0
            is_active = amplitude >= self.env.idle_threshold
            
            if mode == 'relay' and is_active:
                # Draw two-hop connection through rBS
                rbs_pos = self.env.rbs
                su_3d = (su_pos[0], 17, su_pos[1])  # SU antenna top height
                rbs_3d = (rbs_pos[0], 34, rbs_pos[1])  # rBS top height
                du_3d = (du_pos[0], 13, du_pos[1])  # DU antenna top height
                
                self.draw_connection_line(su_3d, rbs_3d, sinr, 'relay', is_active)
                self.draw_connection_line(rbs_3d, du_3d, sinr, 'relay', is_active)
                self.draw_signal_cone(su_3d, rbs_3d, sinr, 'relay', is_active)
                self.draw_signal_cone(rbs_3d, du_3d, sinr, 'relay', is_active)
            else:
                # Direct D2D connection
                su_3d = (su_pos[0], 17, su_pos[1])  # SU antenna top height
                du_3d = (du_pos[0], 13, du_pos[1])  # DU antenna top height
                
                self.draw_connection_line(su_3d, du_3d, sinr, 'd2d', is_active)
                self.draw_signal_cone(su_3d, du_3d, sinr, 'd2d', is_active)
            
            # Draw interference indicator at DU
            if self.show_interference:
                # Approximate interference from other agents
                interference = 0.0
                for k in range(self.env.num_agents):
                    if k != i and k < len(amplitudes):
                        amp_k = amplitudes[k]
                        if amp_k >= self.env.idle_threshold:
                            dist = np.linalg.norm(self.env.sus[k] - du_pos)
                            interference += amp_k / (dist + 1e-6)
                
                # Draw at DU position with proper Y coordinate
                du_pos_3d = (du_pos[0], du_pos[1], 13)  # Note: swapped to match interference function
                self.draw_interference_indicator(du_pos_3d, interference / 10.0)
    
    def update_paths(self):
        """Update movement trails"""
        for i in range(self.env.num_agents):
            # Update SU path
            current_su_pos = tuple(self.env.sus[i])
            if not self.su_paths[i] or \
               np.linalg.norm(np.array(current_su_pos) - np.array(self.su_paths[i][-1])) > 3.0:
                self.su_paths[i].append(current_su_pos)
                if len(self.su_paths[i]) > self.max_path_length:
                    self.su_paths[i].pop(0)
            
            # Update DU path
            current_du_pos = tuple(self.env.dus[i])
            if not self.du_paths[i] or \
               np.linalg.norm(np.array(current_du_pos) - np.array(self.du_paths[i][-1])) > 3.0:
                self.du_paths[i].append(current_du_pos)
                if len(self.du_paths[i]) > self.max_path_length:
                    self.du_paths[i].pop(0)
    
    def draw_3d_scene(self):
        """Draw the complete 3D scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.apply()
        
        # Draw in order (back to front for transparency)
        self.draw_ground()
        self.draw_paths()
        self.draw_relay_station()
        self.draw_jammer()
        self.draw_connections_and_signals()
        self.draw_agents()
        self.draw_destinations()
    
    def draw_2d_overlay(self):
        """Draw 2D UI overlay with metrics and info"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window_width, self.window_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Get environment metrics
        metrics = self.env.get_metrics()
        
        # Main info panel
        info_lines = [
            f"Anti-Jamming Communication System {'[PAUSED]' if self.paused else ''}",
            f"Episode: {self.episode_count} | Step: {self.step_count}",
            "",
            f"Performance Metrics:",
            f"  Success Rate:     {metrics['success_rate']:.2%}",
            f"  Avg Throughput:   {metrics['avg_throughput']:.3f} bits/s/Hz",
            f"  Energy Efficiency: {metrics['energy_efficiency']:.3f}",
            f"  Throughput/Energy: {metrics['throughput_efficiency']:.3f}",
            "",
            f"Environment Stats:",
            f"  Agents (SUs):     {self.env.num_agents}",
            f"  Jammer Power:     {self.env.jammer_power:.2f} W",
            f"  PGA Max Gain:     {self.env.pga_gain}x",
            f"  SINR Threshold:   {self.env.sinr_threshold:.2f}",
            "",
            f"Controls:",
            f"  P: Pause  |  Space: Step  |  R: Reset",
            f"  C: Connections  |  S: Signal Cones",
            f"  I: Interference  |  T: Paths",
            f"  Tab: Select Agent  |  ESC: Exit",
            f"  Mouse: Rotate/Pan/Zoom"
        ]
        
        # Draw main panel background
        panel_height = len(info_lines) * 18 + 20
        glColor4f(0.1, 0.1, 0.15, 0.85)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(350, 10)
        glVertex2f(350, panel_height)
        glVertex2f(10, panel_height)
        glEnd()
        
        # Render text
        y_offset = 22
        for line in info_lines:
            if line == info_lines[0]:
                text_surface = self.font_large.render(line, True, (220, 220, 240))
            elif line.startswith("  "):
                text_surface = self.font_small.render(line, True, (180, 190, 200))
            else:
                text_surface = self.font_medium.render(line, True, (200, 210, 220))
            
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glRasterPos2f(20, y_offset)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            y_offset += 18
        
        # Selected agent info panel
        if self.selected_agent is not None and self.last_info:
            agent_lines = [
                f"Agent {self.selected_agent} Details:",
                "",
                f"Position (SU): ({self.env.sus[self.selected_agent][0]:.1f}, {self.env.sus[self.selected_agent][1]:.1f})",
                f"Position (DU): ({self.env.dus[self.selected_agent][0]:.1f}, {self.env.dus[self.selected_agent][1]:.1f})",
                ""
            ]
            
            if self.selected_agent < len(self.last_info.get('sinr_values', [])):
                sinr = self.last_info['sinr_values'][self.selected_agent]
                mode = self.last_info['modes'][self.selected_agent]
                amplitude = self.last_info['amplitudes'][self.selected_agent]
                pga = self.last_info['pga_choices'][self.selected_agent]
                energy = self.last_info['energy'][self.selected_agent]
                
                agent_lines.extend([
                    f"SINR:       {sinr:.4f} ({10*np.log10(sinr + 1e-9):.2f} dB)",
                    f"Mode:       {mode.upper()}",
                    f"Amplitude:  {amplitude:.3f}",
                    f"PGA Gain:   {pga:.1f}x",
                    f"Energy:     {energy:.4f}",
                    f"Status:     {'TRANSMITTING' if amplitude >= self.env.idle_threshold else 'IDLE'}"
                ])
            
            # Draw agent panel background
            agent_panel_height = len(agent_lines) * 18 + 20
            glColor4f(0.15, 0.1, 0.15, 0.85)
            glBegin(GL_QUADS)
            glVertex2f(10, panel_height + 20)
            glVertex2f(350, panel_height + 20)
            glVertex2f(350, panel_height + 20 + agent_panel_height)
            glVertex2f(10, panel_height + 20 + agent_panel_height)
            glEnd()
            
            # Render agent info text
            y_offset = panel_height + 32
            for line in agent_lines:
                if line == agent_lines[0]:
                    text_surface = self.font_medium.render(line, True, (255, 255, 150))
                else:
                    text_surface = self.font_small.render(line, True, (220, 220, 180))
                
                text_data = pygame.image.tostring(text_surface, "RGBA", True)
                glRasterPos2f(20, y_offset)
                glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                            GL_RGBA, GL_UNSIGNED_BYTE, text_data)
                y_offset += 18
        
        # Restore 3D
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def handle_mouse(self):
        """Handle mouse input for camera control"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        
        if mouse_buttons[0]:  # Left button - rotate
            if not self.mouse_dragging or self.mouse_button != 0:
                self.last_mouse_pos = mouse_pos
                self.mouse_dragging = True
                self.mouse_button = 0
            else:
                dx = (mouse_pos[0] - self.last_mouse_pos[0]) * 0.01
                dy = (mouse_pos[1] - self.last_mouse_pos[1]) * 0.01
                self.camera.rotate(dx, dy)
                self.last_mouse_pos = mouse_pos
        
        elif mouse_buttons[2]:  # Right button - pan
            if not self.mouse_dragging or self.mouse_button != 2:
                self.last_mouse_pos = mouse_pos
                self.mouse_dragging = True
                self.mouse_button = 2
            else:
                dx = (mouse_pos[0] - self.last_mouse_pos[0]) * 0.3
                dy = (mouse_pos[1] - self.last_mouse_pos[1]) * 0.3
                self.camera.pan(-dx, dy)
                self.last_mouse_pos = mouse_pos
        else:
            self.mouse_dragging = False
    
    def step_simulation(self):
        """Execute one simulation step"""
        # Get states
        states = self.env._get_states()
        
        # Select actions (from agent or random)
        if self.agent:
            actions = [self.agent.select_action(states[i], add_noise=False) 
                      for i in range(self.env.num_agents)]
        else:
            # Random actions for demo
            actions = [np.random.uniform(-1, 1, 2) for _ in range(self.env.num_agents)]
        
        # Step environment
        next_states, rewards, dones, info = self.env.step(actions)
        
        # Store info for visualization
        self.last_info = info
        self.step_count += 1
        
        # Update paths
        self.update_paths()
        
        # Check if episode is done (you can add episode termination logic)
        # For now, episodes are infinite
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.env.reset()
        self.su_paths = {i: [] for i in range(self.env.num_agents)}
        self.du_paths = {i: [] for i in range(self.env.num_agents)}
        self.step_count = 0
        self.episode_count += 1
        self.last_info = {}
        self.animation_time = 0.0
    
    def run(self):
        """Main visualization loop"""
        running = True
        
        print("="*80)
        print("3D VISUALIZER RUNNING")
        print("="*80)
        print("Controls:")
        print("  P: Pause/Unpause")
        print("  Space: Step (when paused)")
        print("  R: Reset episode")
        print("  C: Toggle connections")
        print("  S: Toggle signal cones")
        print("  I: Toggle interference visualization")
        print("  T: Toggle movement paths")
        print("  Tab: Select next agent")
        print("  Mouse Left: Rotate camera")
        print("  Mouse Right: Pan camera")
        print("  Mouse Wheel: Zoom")
        print("  ESC: Exit")
        print("="*80)
        
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_SPACE:
                        if self.paused:
                            self.step_simulation()
                    elif event.key == pygame.K_r:
                        self.reset_simulation()
                    elif event.key == pygame.K_c:
                        self.show_connections = not self.show_connections
                    elif event.key == pygame.K_s:
                        self.show_signal_cones = not self.show_signal_cones
                    elif event.key == pygame.K_i:
                        self.show_interference = not self.show_interference
                    elif event.key == pygame.K_t:
                        self.show_paths = not self.show_paths
                    elif event.key == pygame.K_TAB:
                        if self.selected_agent is None:
                            self.selected_agent = 0
                        else:
                            self.selected_agent = (self.selected_agent + 1) % self.env.num_agents
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up
                        self.camera.zoom(-30)
                    elif event.button == 5:  # Scroll down
                        self.camera.zoom(30)
            
            # Handle mouse dragging
            self.handle_mouse()
            
            # Step simulation if not paused
            if not self.paused:
                self.step_simulation()
            
            # Render scene
            self.draw_3d_scene()
            self.draw_2d_overlay()
            
            pygame.display.flip()
            
            # Update animation time
            if not self.paused:
                dt = self.clock.tick(self.render_fps) / 1000.0
                self.animation_time += dt
            else:
                self.clock.tick(self.render_fps)
        
        pygame.quit()
        print("\n✓ Visualizer closed")


def main():
    parser = argparse.ArgumentParser(description='3D Visualizer for Anti-Jamming Communication')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained MADDPG checkpoint')
    parser.add_argument('--fps', type=int, default=60,
                       help='Rendering FPS (default: 60)')
    parser.add_argument('--width', type=int, default=1600,
                       help='Window width (default: 1600)')
    parser.add_argument('--height', type=int, default=900,
                       help='Window height (default: 900)')
    
    args = parser.parse_args()
    
    # Create environment
    print("Initializing environment...")
    env = CommEnvironment(
        num_agents=5,
        mutual_interf_coef=0.06,
        idle_threshold=0.18,
        reward_w_thr=4.0,
        reward_w_succ=1.2,
        reward_w_margin=0.8,
        reward_w_energy=0.05
    )
    
    # Load agent if checkpoint provided
    agent = None
    if args.checkpoint:
        print(f"Loading trained agent from {args.checkpoint}...")
        from maddpg import MADDPGAgent
        import torch
        
        agents = [MADDPGAgent(env.state_dim, env.action_dim, i, env.num_agents)
                 for i in range(env.num_agents)]
        
        # Load checkpoint (assuming it's a list of agent states)
        checkpoint = torch.load(args.checkpoint)
        for i, agent_state in enumerate(checkpoint):
            agents[i].actor.load_state_dict(agent_state['actor_state_dict'])
            agents[i].actor.eval()
        
        # Use first agent for demo (in practice, you'd use all agents)
        agent = agents[0]
        print("✓ Agent loaded")
    else:
        print("No checkpoint provided - using random actions")
    
    # Create and run visualizer
    visualizer = CommVisualizer3D(
        env=env,
        agent=agent,
        window_width=args.width,
        window_height=args.height,
        render_fps=args.fps
    )
    
    visualizer.run()


if __name__ == "__main__":
    main()