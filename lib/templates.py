from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jinja2.meta
from jinja2 import Template as Template, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


@dataclass
class TemplateWrapper:
    template: str
    constants: dict[str, str]

    def render(self, **kwargs) -> str:
        kwargs = {**self.constants, **kwargs}
        template = compile(self.template)
        return template.render(**kwargs)


def load_chat_template(
        file: str,
        strip: bool = True,
        bos_token: str = '<s>',
        eos_token: str = '</s>',
        add_generation_prompt: bool = True,
        **constants: dict[str, str],
) -> TemplateWrapper:
    if bos_token is not None:
        constants['bos_token'] = bos_token

    if eos_token is not None:
        constants['eos_token'] = eos_token

    if add_generation_prompt is not None:
        constants['add_generation_prompt'] = add_generation_prompt

    return load_template(file, strip=strip, **constants)


def load_template(
        file: str,
        strip: bool = False,
        **constants: dict[str, str],
) -> TemplateWrapper:
    template = load(file, strip=strip)
    return TemplateWrapper(template=template, constants=constants)


def load(file: str, strip: bool = True) -> Template:
    with open(file) as fp:
        code = fp.read()

    _ = compile(code, cache=False)
    code = code.strip()
    if strip:
        code = strip_whitespace(code)

    return code


def compile(template: str, cache: bool = True) -> Template:
    """Convert a string to a ``jinja2.Template``.

    Results can be cached to prevent repeated recompilation.

    Duplicates logic from Hugging Face.
    https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L1763

    The only difference from the Hugging Face code is explicitly
    passing ``autoescape=False`` when instantiating the
    ``ImmutableSandboxedEnvironment``. The current default in jinja2
    is already ``autoescape=False``, but the Jinja2 docs recommend
    being explicit about this.

        In future versions of Jinja we might enable autoescaping by default
        for security reasons. As such you are encouraged to explicitly
        configure autoescaping now instead of relying on the default.

    docs: https://jinja.palletsprojects.com/en/3.1.x/api/
    """
    if cache:
        return _compile_with_cache(template)
    else:
        return _compile_without_cache(template)


def strip_whitespace(template: str) -> str:
    """Strip leading and trailing whitespace from all lines in a string."""
    return ''.join(line.strip() for line in template.split('\n'))


@lru_cache
def _compile_with_cache(template: str) -> Template:
    """Convert a string to a ``jinja2.Template``."""
    return _compile_without_cache(template)


def _compile_without_cache(template: str) -> Template:
    """Convert a string to a ``jinja2.Template``."""
    env = create_sandbox_environment()
    return env.from_string(template)


def find_undeclared_variables(template: str) -> list[str]:
    """Find any undeclared variables according to Jinja.

    The result is a little noisy, but is useful for quick sanity checks.
    """
    env = create_sandbox_environment()
    ast = env.parse(template)
    return sorted(jinja2.meta.find_undeclared_variables(ast))


def create_sandbox_environment() -> ImmutableSandboxedEnvironment:
    """Create a sandbox jinja environment."""

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals['raise_exception'] = raise_exception
    return env
