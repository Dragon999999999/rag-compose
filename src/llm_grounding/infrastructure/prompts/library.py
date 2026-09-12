"""Load versioned Jinja prompt templates from the repository prompt library.

Prompt files define model-facing behavior while this module supplies safe template
selection, metadata, and runtime rendering. The checked-in top-level `prompts/`
directory is used in editable checkouts; wheel builds bundle the same files as
package assets.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError


class PromptLibraryError(ValueError):
    """Report invalid or unavailable prompt-library resources."""


@dataclass(frozen=True)
class PromptMetadata:
    """Identify one registered versioned prompt template."""

    prompt_id: str
    version: str
    template: str


@dataclass(frozen=True)
class RenderedPrompt:
    """Pair rendered model input with the prompt metadata that produced it."""

    metadata: PromptMetadata
    text: str


class PromptLibrary:
    """Resolve registered prompt IDs and render their Jinja templates."""

    def __init__(self, root: Path) -> None:
        """Load the registry and create a strict template environment.

        Args:
            root: Directory containing `registry.yaml` and referenced templates.

        Raises:
            PromptLibraryError: If the registry or a template path is invalid.
        """
        self._root = root.resolve()
        self._metadata = self._load_registry()
        self._environment = Environment(
            loader=FileSystemLoader(self._root),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )

    @classmethod
    def default(cls) -> "PromptLibrary":
        """Load repository prompts in source trees or bundled wheel assets."""
        checkout_root = Path(__file__).resolve().parents[4]
        checkout_prompts = checkout_root / "prompts"
        if checkout_prompts.is_dir():
            return cls(checkout_prompts)
        assets = Path(str(files("llm_grounding.infrastructure.prompts") / "assets"))
        return cls(assets)

    def metadata(self, prompt_id: str) -> PromptMetadata:
        """Return registered metadata for a configured prompt ID."""
        try:
            return self._metadata[prompt_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._metadata)) or "none"
            raise PromptLibraryError(
                f"Unknown prompt ID '{prompt_id}'; available IDs: {available}."
            ) from exc

    def render(self, prompt_id: str, variables: Mapping[str, object]) -> RenderedPrompt:
        """Render one registered prompt with strict runtime-variable validation."""
        metadata = self.metadata(prompt_id)
        try:
            template = self._environment.get_template(metadata.template)
            text = template.render(**variables)
        except TemplateError as exc:
            raise PromptLibraryError(f"Unable to render prompt '{prompt_id}'.") from exc
        return RenderedPrompt(metadata, text)

    def _load_registry(self) -> dict[str, PromptMetadata]:
        """Validate registry entries and prevent template paths leaving the library."""
        registry_path = self._root / "registry.yaml"
        try:
            content = registry_path.read_text(encoding="utf-8")
            document = yaml.safe_load(content)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise PromptLibraryError("Unable to read prompt registry.") from exc
        if not isinstance(document, dict) or not isinstance(
            document.get("prompts"), dict
        ):
            raise PromptLibraryError("Prompt registry must define a prompts mapping.")
        metadata: dict[str, PromptMetadata] = {}
        for prompt_id, entry in document["prompts"].items():
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                raise PromptLibraryError("Prompt IDs must be nonempty strings.")
            if not isinstance(entry, dict):
                raise PromptLibraryError(
                    "Each prompt registry entry must be a mapping."
                )
            version, template = entry.get("version"), entry.get("template")
            if not isinstance(version, str) or not version.strip():
                raise PromptLibraryError("Prompt versions must be nonempty strings.")
            if not isinstance(template, str) or not template.strip():
                raise PromptLibraryError("Prompt templates must be nonempty paths.")
            template_path = (self._root / template).resolve()
            if self._root not in template_path.parents or not template_path.is_file():
                raise PromptLibraryError(
                    f"Prompt template for '{prompt_id}' is unavailable."
                )
            metadata[prompt_id] = PromptMetadata(prompt_id, version, template)
        return metadata
