"""
Compatibility helpers for importing legacy timm releases on Python 3.12+.

Older timm versions define dataclass instances as defaults in model config
classes. Python 3.12 started rejecting unhashable defaults during dataclass
processing, which breaks timm import before the repo code runs. Upgrading timm
is the preferred fix, but this targeted patch keeps the existing environment
working without touching site-packages.
"""

from __future__ import annotations

import dataclasses
import sys
import types


def patch_timm_dataclass_defaults() -> None:
    if sys.version_info < (3, 12):
        return
    if getattr(dataclasses, "_timm_py312_compat_applied", False):
        return

    MISSING = dataclasses.MISSING
    Field = dataclasses.Field
    field = dataclasses.field
    _FIELD = dataclasses._FIELD
    _FIELD_CLASSVAR = dataclasses._FIELD_CLASSVAR
    _FIELD_INITVAR = dataclasses._FIELD_INITVAR
    _is_classvar = dataclasses._is_classvar
    _is_type = dataclasses._is_type
    _is_initvar = dataclasses._is_initvar

    def _get_field_timm_compat(cls, a_name, a_type, default_kw_only):
        default = getattr(cls, a_name, MISSING)
        if isinstance(default, Field):
            f = default
        else:
            if isinstance(default, types.MemberDescriptorType):
                default = MISSING
            f = field(default=default)

        f.name = a_name
        f.type = a_type
        f._field_type = _FIELD

        typing = sys.modules.get("typing")
        if typing:
            if _is_classvar(a_type, typing) or (
                isinstance(f.type, str)
                and _is_type(f.type, cls, typing, typing.ClassVar, _is_classvar)
            ):
                f._field_type = _FIELD_CLASSVAR

        if f._field_type is _FIELD:
            dataclasses_mod = sys.modules["dataclasses"]
            if _is_initvar(a_type, dataclasses_mod) or (
                isinstance(f.type, str)
                and _is_type(
                    f.type, cls, dataclasses_mod, dataclasses_mod.InitVar, _is_initvar
                )
            ):
                f._field_type = _FIELD_INITVAR

        if f._field_type in (_FIELD_CLASSVAR, _FIELD_INITVAR):
            if f.default_factory is not MISSING:
                raise TypeError(f"field {f.name} cannot have a default factory")

        if f._field_type in (_FIELD, _FIELD_INITVAR):
            if f.kw_only is MISSING:
                f.kw_only = default_kw_only
        else:
            assert f._field_type is _FIELD_CLASSVAR
            if f.kw_only is not MISSING:
                raise TypeError(f"field {f.name} is a ClassVar but specifies kw_only")

        if f._field_type is _FIELD and f.default.__class__.__hash__ is None:
            default_module = type(f.default).__module__
            if not default_module.startswith("timm."):
                raise ValueError(
                    f"mutable default {type(f.default)} for field {f.name} "
                    "is not allowed: use default_factory"
                )

        return f

    dataclasses._get_field = _get_field_timm_compat
    dataclasses._timm_py312_compat_applied = True
