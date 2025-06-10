import numpy as np
import os
import ctypes
from typing import Optional, Tuple, Union
from pathlib import Path

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Limited texture support.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import warp as wp

class TextureLoader:
    """OpenGL texture loader with support for various image formats"""
    
    # Supported formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tga', '.tiff', '.tif', '.hdr', '.exr'}
    
    @staticmethod
    def load_image_pil(filepath: str) -> Tuple[np.ndarray, int, int, int]:
        """Load image using PIL/Pillow"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for this image format")
        
        try:
            img = Image.open(filepath)
            
            # Convert to RGB if needed
            if img.mode == 'RGBA':
                channels = 4
            elif img.mode == 'RGB':
                channels = 3
            elif img.mode in ['L', 'P']:
                img = img.convert('RGB')
                channels = 3
            else:
                img = img.convert('RGB')
                channels = 3
            
            # Flip image vertically for OpenGL coordinate system
            img = ImageOps.flip(img)
            
            # Convert to numpy array
            img_data = np.array(img, dtype=np.uint8)
            height, width = img_data.shape[:2]
            
            return img_data, width, height, channels
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {filepath}: {e}")
    
    @staticmethod
    def load_image_cv2(filepath: str) -> Tuple[np.ndarray, int, int, int]:
        """Load image using OpenCV"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for this image format")
        
        try:
            # Load image
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image: {filepath}")
            
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            
            # Convert BGR to RGB for OpenGL
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            
            # Flip image vertically for OpenGL coordinate system
            img = cv2.flip(img, 0)
            
            return img, width, height, channels
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {filepath}: {e}")
    
    @staticmethod
    def create_solid_color_texture(color: Tuple[float, float, float] = (1.0, 1.0, 1.0), 
                                 size: int = 1) -> Tuple[np.ndarray, int, int, int]:
        """Create a solid color texture"""
        color_255 = [int(c * 255) for c in color]
        img_data = np.full((size, size, 3), color_255, dtype=np.uint8)
        return img_data, size, size, 3
    
    @staticmethod
    def create_checkerboard_texture(color1: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                                  color2: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                                  size: int = 256,
                                  checker_size: int = 32) -> Tuple[np.ndarray, int, int, int]:
        """Create a checkerboard texture"""
        img_data = np.zeros((size, size, 3), dtype=np.uint8)
        
        color1_255 = [int(c * 255) for c in color1]
        color2_255 = [int(c * 255) for c in color2]
        
        for i in range(size):
            for j in range(size):
                checker_x = i // checker_size
                checker_y = j // checker_size
                if (checker_x + checker_y) % 2 == 0:
                    img_data[i, j] = color1_255
                else:
                    img_data[i, j] = color2_255
        
        return img_data, size, size, 3
    
    @staticmethod
    def create_grid_texture(line_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                          bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                          size: int = 256,
                          grid_size: int = 32,
                          line_width: int = 2) -> Tuple[np.ndarray, int, int, int]:
        """Create a grid texture"""
        img_data = np.full((size, size, 3), [int(c * 255) for c in bg_color], dtype=np.uint8)
        line_color_255 = [int(c * 255) for c in line_color]
        
        # Draw horizontal lines
        for i in range(0, size, grid_size):
            for w in range(line_width):
                if i + w < size:
                    img_data[i + w, :] = line_color_255
        
        # Draw vertical lines
        for j in range(0, size, grid_size):
            for w in range(line_width):
                if j + w < size:
                    img_data[:, j + w] = line_color_255
        
        return img_data, size, size, 3

class OpenGLTextureManager:
    """Manages OpenGL textures with caching and automatic format handling"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        self.textures = {}  # filepath -> texture_id
        self.texture_cache = {}  # texture_id -> metadata
        self._next_id = 1
        
        # Initialize OpenGL if needed
        if hasattr(renderer, 'gl'):
            self.gl = renderer.gl
        else:
            try:
                from pyglet import gl
                self.gl = gl
            except ImportError:
                raise ImportError("OpenGL context not available")
    
    def load_texture(self, filepath: Union[str, Path], 
                    generate_mipmaps: bool = True,
                    wrap_s: int = None,
                    wrap_t: int = None,
                    min_filter: int = None,
                    mag_filter: int = None) -> int:
        """
        Load a texture from file and return OpenGL texture ID
        
        Args:
            filepath: Path to the image file
            generate_mipmaps: Whether to generate mipmaps
            wrap_s: Texture wrapping mode for S coordinate
            wrap_t: Texture wrapping mode for T coordinate
            min_filter: Minification filter
            mag_filter: Magnification filter
            
        Returns:
            OpenGL texture ID
        """
        filepath = str(filepath)
        
        # Check cache first
        if filepath in self.textures:
            return self.textures[filepath]
        
        # Set default parameters
        if wrap_s is None:
            wrap_s = self.gl.GL_REPEAT
        if wrap_t is None:
            wrap_t = self.gl.GL_REPEAT
        if min_filter is None:
            min_filter = self.gl.GL_LINEAR_MIPMAP_LINEAR if generate_mipmaps else self.gl.GL_LINEAR
        if mag_filter is None:
            mag_filter = self.gl.GL_LINEAR
        
        # Load image data
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Texture file not found: {filepath}")
        
        ext = Path(filepath).suffix.lower()
        if ext not in TextureLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported texture format: {ext}")
        
        # Try PIL first, then OpenCV
        try:
            img_data, width, height, channels = TextureLoader.load_image_pil(filepath)
        except:
            try:
                img_data, width, height, channels = TextureLoader.load_image_cv2(filepath)
            except Exception as e:
                raise RuntimeError(f"Failed to load texture {filepath}: {e}")
        
        # Create OpenGL texture
        texture_id = self._create_gl_texture(img_data, width, height, channels,
                                           generate_mipmaps, wrap_s, wrap_t, 
                                           min_filter, mag_filter)
        
        # Cache the texture
        self.textures[filepath] = texture_id
        self.texture_cache[texture_id] = {
            'filepath': filepath,
            'width': width,
            'height': height,
            'channels': channels,
            'mipmaps': generate_mipmaps
        }
        
        return texture_id
    
    def create_procedural_texture(self, texture_type: str, **kwargs) -> int:
        """
        Create a procedural texture
        
        Args:
            texture_type: Type of procedural texture ('solid', 'checkerboard', 'grid')
            **kwargs: Parameters for the texture generation
            
        Returns:
            OpenGL texture ID
        """
        cache_key = f"{texture_type}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key in self.textures:
            return self.textures[cache_key]
        
        if texture_type == 'solid':
            img_data, width, height, channels = TextureLoader.create_solid_color_texture(**kwargs)
        elif texture_type == 'checkerboard':
            img_data, width, height, channels = TextureLoader.create_checkerboard_texture(**kwargs)
        elif texture_type == 'grid':
            img_data, width, height, channels = TextureLoader.create_grid_texture(**kwargs)
        else:
            raise ValueError(f"Unknown procedural texture type: {texture_type}")
        
        texture_id = self._create_gl_texture(img_data, width, height, channels)
        
        self.textures[cache_key] = texture_id
        self.texture_cache[texture_id] = {
            'filepath': cache_key,
            'width': width,
            'height': height,
            'channels': channels,
            'mipmaps': True
        }
        
        return texture_id
    
    def _create_gl_texture(self, img_data: np.ndarray, width: int, height: int, channels: int,
                          generate_mipmaps: bool = True,
                          wrap_s: int = None, wrap_t: int = None,
                          min_filter: int = None, mag_filter: int = None) -> int:
        """Create OpenGL texture from image data"""
        
        # Ensure we're in the right OpenGL context
        if hasattr(self.renderer, '_switch_context'):
            self.renderer._switch_context()
        
        # Set defaults
        if wrap_s is None:
            wrap_s = self.gl.GL_REPEAT
        if wrap_t is None:
            wrap_t = self.gl.GL_REPEAT
        if min_filter is None:
            min_filter = self.gl.GL_LINEAR_MIPMAP_LINEAR if generate_mipmaps else self.gl.GL_LINEAR
        if mag_filter is None:
            mag_filter = self.gl.GL_LINEAR
        
        # Generate texture
        texture = self.gl.GLuint()
        self.gl.glGenTextures(1, texture)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, texture)
        
        # Determine format
        if channels == 1:
            internal_format = self.gl.GL_RED
            format_type = self.gl.GL_RED
        elif channels == 3:
            internal_format = self.gl.GL_RGB
            format_type = self.gl.GL_RGB
        elif channels == 4:
            internal_format = self.gl.GL_RGBA
            format_type = self.gl.GL_RGBA
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
        
        # Upload texture data
        self.gl.glTexImage2D(
            self.gl.GL_TEXTURE_2D, 0, internal_format,
            width, height, 0, format_type,
            self.gl.GL_UNSIGNED_BYTE, img_data.ctypes.data
        )
        
        # Set texture parameters
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_S, wrap_s)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_T, wrap_t)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, min_filter)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, mag_filter)
        
        # Generate mipmaps if requested
        if generate_mipmaps:
            self.gl.glGenerateMipmap(self.gl.GL_TEXTURE_2D)
        
        # Unbind texture
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)
        
        return int(texture.value)
    
    def bind_texture(self, texture_id: int, unit: int = 0):
        """Bind texture to specified texture unit"""
        self.gl.glActiveTexture(self.gl.GL_TEXTURE0 + unit)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, texture_id)
    
    def unbind_texture(self, unit: int = 0):
        """Unbind texture from specified texture unit"""
        self.gl.glActiveTexture(self.gl.GL_TEXTURE0 + unit)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)
    
    def delete_texture(self, texture_id: int):
        """Delete a texture and remove from cache"""
        if texture_id in self.texture_cache:
            filepath = self.texture_cache[texture_id]['filepath']
            if filepath in self.textures:
                del self.textures[filepath]
            del self.texture_cache[texture_id]
        
        texture = self.gl.GLuint(texture_id)
        self.gl.glDeleteTextures(1, texture)
    
    def get_texture_info(self, texture_id: int) -> dict:
        """Get information about a loaded texture"""
        return self.texture_cache.get(texture_id, {})
    
    def clear_cache(self):
        """Clear all cached textures"""
        for texture_id in list(self.texture_cache.keys()):
            self.delete_texture(texture_id)
        self.textures.clear()
        self.texture_cache.clear()
    
    def __del__(self):
        """Cleanup textures when manager is destroyed"""
        try:
            self.clear_cache()
        except:
            pass  # Ignore errors during cleanup

# Convenience functions for easy texture loading
def load_texture(renderer, filepath: str, **kwargs) -> int:
    """Convenience function to load a single texture"""
    if not hasattr(renderer, '_texture_manager'):
        renderer._texture_manager = OpenGLTextureManager(renderer)
    return renderer._texture_manager.load_texture(filepath, **kwargs)

def create_checkerboard_texture(renderer, color1=(1.0, 1.0, 1.0), color2=(0.5, 0.5, 0.5), size=256) -> int:
    """Convenience function to create a checkerboard texture"""
    if not hasattr(renderer, '_texture_manager'):
        renderer._texture_manager = OpenGLTextureManager(renderer)
    return renderer._texture_manager.create_procedural_texture(
        'checkerboard', color1=color1, color2=color2, size=size
    )

def create_solid_texture(renderer, color=(1.0, 1.0, 1.0), size=1) -> int:
    """Convenience function to create a solid color texture"""
    if not hasattr(renderer, '_texture_manager'):
        renderer._texture_manager = OpenGLTextureManager(renderer)
    return renderer._texture_manager.create_procedural_texture(
        'solid', color=color, size=size
    )

def create_grid_texture(renderer, line_color=(0.0, 0.0, 0.0), bg_color=(1.0, 1.0, 1.0), size=256) -> int:
    """Convenience function to create a grid texture"""
    if not hasattr(renderer, '_texture_manager'):
        renderer._texture_manager = OpenGLTextureManager(renderer)
    return renderer._texture_manager.create_procedural_texture(
        'grid', line_color=line_color, bg_color=bg_color, size=size
    )