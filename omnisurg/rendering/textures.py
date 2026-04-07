from newton._src.utils.texture import compute_texture_hash

_PATCHED = False


def enable_persistent_gl_textures():
    """Avoid re-uploading unchanged diffuse maps on every deforming-mesh update."""

    global _PATCHED
    if _PATCHED:
        return

    from newton._src.viewer.gl import opengl

    if getattr(opengl.MeshGL.update_texture, "__omnisurg_patched__", False):
        _PATCHED = True
        return

    original_update_texture = opengl.MeshGL.update_texture

    def cached_update_texture(self, texture=None):
        texture_hash = compute_texture_hash(texture)
        if getattr(self, "_omnisurg_texture_hash", None) == texture_hash:
            return
        original_update_texture(self, texture)
        self._omnisurg_texture_hash = texture_hash

    cached_update_texture.__omnisurg_patched__ = True
    opengl.MeshGL.update_texture = cached_update_texture
    _PATCHED = True
