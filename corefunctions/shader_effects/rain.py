"""
Complete rain effect - rendering + event integration
Everything needed for rain in one place!
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_rain(state, outstate, intensity=1.0, wind=0.0):
    """
    Shader-based rain effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_rain, intensity=1.5, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        intensity: Rain intensity multiplier (affects number of drops)
        wind: Wind effect (-1 to 1, affects drop angle)
    """
    # Get the viewport
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize rain effect on first call
    if state['count'] == 0:
        num_drops = int(100 * intensity)
        print(f"Initializing rain effect for frame {frame_id} with {num_drops} drops")
        
        try:
            rain_effect = viewport.add_effect(
                RainEffect,
                num_raindrops=num_drops,
                wind=wind
            )
            state['rain_effect'] = rain_effect
            print(f"✓ Initialized shader rain for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize rain: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update wind if it changes in global state
    if 'rain_effect' in state:
        state['rain_effect'].wind = outstate.get('wind', wind)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'rain_effect' in state:
            print(f"Cleaning up rain effect for frame {frame_id}")
            viewport.effects.remove(state['rain_effect'])
            state['rain_effect'].cleanup()
            print(f"✓ Cleaned up shader rain for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class RainEffect(ShaderEffect):
    """GPU-based rain effect using instanced rendering with vectorized updates"""
    
    def __init__(self, viewport, num_raindrops: int = 100, wind: float = 0.0):
        super().__init__(viewport)
        self.num_raindrops = num_raindrops
        self.base_num_raindrops = num_raindrops
        self.wind = wind
        self.instance_VBO = None
        
        # Vectorized raindrop data (all stored as numpy arrays)
        self.positions = None  # [x, y, z] - shape (N, 3)
        self.velocities = None  # [speed] - shape (N,)
        self.base_velocities = None  # Original speeds for intensity scaling
        self.dimensions = None  # [width, length] - shape (N, 2)
        self.alphas = None  # [alpha] - shape (N,)
        self.colors = None  # [r, g, b] - shape (N, 3)
        
        self._initialize_raindrops()
        
    def _initialize_raindrops(self):
        """Initialize all raindrop data as numpy arrays"""
        n = self.num_raindrops
        
        # Positions: x, y, z
        self.positions = np.column_stack([
            np.random.uniform(0, self.viewport.width, n),  # x
            np.random.uniform(0, self.viewport.height, n),  # y (randomized)
            np.random.uniform(0, 100, n)  # z (depth)
        ])
        
        # Velocities
        self.velocities = np.random.uniform(100, 300, n)
        self.base_velocities = self.velocities.copy()
        
        # Dimensions based on depth
        depth_factors = self.positions[:, 2] / 100.0  # 0.0 (far) to 1.0 (near)
        base_widths = np.random.uniform(1.0, 2.0, n)
        base_lengths = np.random.uniform(10, 20, n)
        
        self.dimensions = np.column_stack([
            base_widths * (0.3 + 0.7 * depth_factors),
            base_lengths * (0.3 + 0.7 * depth_factors)
        ])
        
        # Alpha based on depth
        self.alphas = 0.2 + 0.6 * depth_factors
        
        # Colors: slight variations of blue/white
        # Random mix of blue (0.3, 0.7, 1.0) to white-ish (0.6, 0.85, 1.0)
        self.colors = np.column_stack([
            np.random.uniform(0.3, 0.7, n),  # R: blue to cyan
            np.random.uniform(0.7, 0.9, n),  # G: consistent
            np.ones(n)  # B: full blue
        ])
        
    def _reset_raindrops(self, mask):
        """Reset raindrops that are off-screen (vectorized)"""
        n_reset = np.sum(mask)
        if n_reset == 0:
            return
            
        # Reset positions
        self.positions[mask, 0] = np.random.uniform(0, self.viewport.width, n_reset)  # x
        self.positions[mask, 1] = -10  # y (top of screen)
        self.positions[mask, 2] = np.random.uniform(0, 100, n_reset)  # z
        
        # Reset velocities
        self.velocities[mask] = np.random.uniform(100, 300, n_reset)
        self.base_velocities[mask] = self.velocities[mask]
        
        # Recalculate dimensions and alpha based on new depth
        depth_factors = self.positions[mask, 2] / 100.0
        base_widths = np.random.uniform(1.0, 2.0, n_reset)
        base_lengths = np.random.uniform(10, 20, n_reset)
        
        self.dimensions[mask, 0] = base_widths * (0.3 + 0.7 * depth_factors)
        self.dimensions[mask, 1] = base_lengths * (0.3 + 0.7 * depth_factors)
        self.alphas[mask] = 0.2 + 0.6 * depth_factors
        
        # Reset colors
        self.colors[mask, 0] = np.random.uniform(0.3, 0.7, n_reset)
        self.colors[mask, 1] = np.random.uniform(0.7, 0.9, n_reset)
        self.colors[mask, 2] = 1.0
        
    def _resize_raindrop_arrays(self, new_size):
        """Resize raindrop arrays when intensity changes"""
        current_size = len(self.positions)
        
        if new_size > current_size:
            # Add new raindrops
            n_new = new_size - current_size
            
            new_positions = np.column_stack([
                np.random.uniform(0, self.viewport.width, n_new),
                np.random.uniform(0, self.viewport.height, n_new),
                np.random.uniform(0, 100, n_new)
            ])
            
            new_velocities = np.random.uniform(100, 300, n_new)
            
            depth_factors = new_positions[:, 2] / 100.0
            base_widths = np.random.uniform(1.0, 2.0, n_new)
            base_lengths = np.random.uniform(10, 20, n_new)
            
            new_dimensions = np.column_stack([
                base_widths * (0.3 + 0.7 * depth_factors),
                base_lengths * (0.3 + 0.7 * depth_factors)
            ])
            
            new_alphas = 0.2 + 0.6 * depth_factors
            
            new_colors = np.column_stack([
                np.random.uniform(0.3, 0.7, n_new),
                np.random.uniform(0.7, 0.9, n_new),
                np.ones(n_new)
            ])
            
            # Concatenate
            self.positions = np.vstack([self.positions, new_positions])
            self.velocities = np.concatenate([self.velocities, new_velocities])
            self.base_velocities = np.concatenate([self.base_velocities, new_velocities])
            self.dimensions = np.vstack([self.dimensions, new_dimensions])
            self.alphas = np.concatenate([self.alphas, new_alphas])
            self.colors = np.vstack([self.colors, new_colors])
            
        elif new_size < current_size:
            # Remove excess raindrops
            self.positions = self.positions[:new_size]
            self.velocities = self.velocities[:new_size]
            self.base_velocities = self.base_velocities[:new_size]
            self.dimensions = self.dimensions[:new_size]
            self.alphas = self.alphas[:new_size]
            self.colors = self.colors[:new_size]
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Centered quad: -0.5 to 0.5
        layout(location = 1) in vec3 offset;  // x, y, z
        layout(location = 2) in vec2 size;
        layout(location = 3) in vec4 color;
        layout(location = 4) in float rotation;  // rotation angle

        out vec4 fragColor;
        out vec2 vertPos;  // Pass through for gradient (0 to 1)
        uniform vec2 resolution;

        void main() {
            // Convert position from -0.5,0.5 range to 0,1 for gradient
            vertPos = position + 0.5;
            
            // First: Scale the quad
            vec2 scaled = position * size;
            
            // Second: Rotate the scaled quad around origin
            float cosR = cos(rotation);
            float sinR = sin(rotation);
            vec2 rotated = vec2(
                scaled.x * cosR - scaled.y * sinR,
                scaled.x * sinR + scaled.y * cosR
            );
            
            // Third: Translate to final position
            vec2 pos = rotated + offset.xy;
            
            // Convert screen coordinates to clip space
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Use Z for depth buffer (normalize to 0-1 range)
            float depth = offset.z / 100.0;
            
            gl_Position = vec4(clipPos, depth, 1.0);
            fragColor = color;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 vertPos;  // Position within quad (0 to 1)
        out vec4 outColor;

        void main() {
            // Create vertical gradient: fade out at top, full at bottom
            // vertPos.y goes from 0 (bottom) to 1 (top) in our coordinate system
            // We want alpha = 0 at top, alpha = 1 at bottom
            float fade = 1.0 - vertPos.y;
            
            // Apply fade to alpha
            outColor = vec4(fragColor.rgb, fragColor.a * fade);
        }
        """
    
    def compile_shader(self):
        """Compile and link rain shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
            # Set resolution uniform
            glUseProgram(shader)
            loc = glGetUniformLocation(shader, "resolution")
            if loc != -1:
                glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
            glUseProgram(0)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise

    def setup_buffers(self):
        """Initialize OpenGL buffers for instanced rendering"""
        # Sort by Z (far to near) for proper depth rendering
        sort_indices = np.argsort(self.positions[:, 2])
        self.positions = self.positions[sort_indices]
        self.velocities = self.velocities[sort_indices]
        self.base_velocities = self.base_velocities[sort_indices]
        self.dimensions = self.dimensions[sort_indices]
        self.alphas = self.alphas[sort_indices]
        self.colors = self.colors[sort_indices]

        # Quad vertices - CENTERED from -0.5 to 0.5 for proper rotation
        # Note: Y axis - negative is bottom, positive is top
        vertices = np.array([
            -0.5, -0.5,  # Bottom left
             0.5, -0.5,  # Bottom right
             0.5,  0.5,  # Top right
            -0.5,  0.5   # Top left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer (quad template)
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Instance buffer (will be updated each frame)
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        """Update raindrop positions (vectorized)"""
        if not self.enabled:
            return
        
        # Get global rain intensity (0.0 to 1.0+)
        rain_intensity = state.get('rain', 1.0)
        
        # Calculate target number of drops
        target_drops = int(self.base_num_raindrops * rain_intensity)
        
        # Resize arrays if needed
        if target_drops != len(self.positions):
            self._resize_raindrop_arrays(target_drops)
            self.num_raindrops = target_drops
        
        # Update velocities based on intensity
        self.velocities = self.base_velocities * (rain_intensity + 0.2)
        
        # Vectorized position updates
        self.positions[:, 1] += self.velocities * dt  # Update y
        self.positions[:, 0] += self.wind * 50 * dt  # Update x (wind)
        
        # Horizontal wrapping
        left_mask = self.positions[:, 0] < -10
        right_mask = self.positions[:, 0] > self.viewport.width + 10
        self.positions[left_mask, 0] = self.viewport.width + 10
        self.positions[right_mask, 0] = -10
        
        # Reset drops that went off bottom
        bottom_mask = self.positions[:, 1] > self.viewport.height + 10
        self._reset_raindrops(bottom_mask)

    def render(self, state: Dict):
        """Render all raindrops using instancing"""
        if not self.enabled or not self.shader or len(self.positions) == 0:
            return
        
        #glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST) 
        glDepthFunc(GL_LESS)  
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        horizontal_velocity = self.wind * 50
        # Vertical velocity component (per raindrop)
        vertical_velocity = self.velocities
        
        # Calculate angle from vertical using velocity components
        # atan2(horizontal, vertical) gives angle from vertical axis
        velocity_angle = np.arctan2(horizontal_velocity, vertical_velocity)
        base_rotation = np.pi   # 90 degrees (to orient vertically)
        rotations = (base_rotation - velocity_angle).astype(np.float32)
        
        # Build instance data (vectorized) - interleave all attributes
        # Note: No fade_alphas calculation here - fade is now per-drop in fragment shader
        instance_data = np.hstack([
            self.positions,  # x, y, z (3 floats)
            self.dimensions,  # width, length (2 floats)
            self.colors,  # r, g, b (3 floats) - per-raindrop colors
            self.alphas[:, np.newaxis],  # alpha (1 float) - base alpha from depth
            rotations[:, np.newaxis]  # rotation (1 float) - wind direction
        ]).astype(np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 10 * 4  # 10 floats * 4 bytes
        
        # Offset (location 1) - vec3
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Size (location 2) - vec2
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color (location 3) - vec4
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Rotation (location 4) - float
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Draw all drops in one call
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_raindrops)
        
        glBindVertexArray(0)
        glUseProgram(0)