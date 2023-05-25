import hashlib
import os
import sys
from datetime import timedelta
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

import pydantic
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from marvin import ai_fn, ai_model
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.artifacts import create_markdown_artifact
from prefect.context import TaskRunContext
from prefect.utilities.hashing import hash_objects
from readability import Document
from slugify import slugify


GPT3 = "gpt-3.5-turbo"
MODEL = os.environ.get("MARVIN_OPENAI_MODEL_NAME", GPT3)


# If we're using GPT 3.5, limit the input to 1500 tokens
# to try and stay under the token limit.
if MODEL == GPT3:
    TOKEN_LIMIT = 1500
else:
    TOKEN_LIMIT = 3000


environment = Environment(loader=FileSystemLoader("./"))
monster_template = environment.get_template("monster.jinja")


class GenerationFailed(Exception):
    """Raised when the model fails to generate a suitable monster."""


def ai_fn_task_input_hash(
    context: "TaskRunContext", arguments: Dict[str, Any]
) -> Optional[str]:
    """
    A variation of `task_input_hash` that hashes the task function's docstring
    instead of the function's code.
    
    AI functions don't have code, and the signal that the function has changed
    is actually in the docstring, where the user writes a GPT prompt.
    """
    fn_doc = context.task.fn.fn.__doc__ or ""
    doc_hash = hashlib.sha256(fn_doc.encode()).hexdigest()

    return hash_objects(
        context.task.task_key,
        doc_hash,
        arguments,
    )


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=3,
)
def get_page(url):
    """Get the HTML page from the URL."""
    return urlopen(url).read()


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=3,
)
def get_text(page):
    """Get the article text from the HTML page."""
    article_html = Document(page).summary()
    article_text = BeautifulSoup(article_html, features="lxml").get_text()
    return article_text[:TOKEN_LIMIT]


@ai_model
class NamedRulesPair(pydantic.BaseModel):
    """A named set of rules."""

    name: str
    rules: str


@ai_model
class Monster(pydantic.BaseModel):
    """A Dungeons and Dragons fifth edition (5e) monster."""

    name: str
    ability_scores: str
    ac: str
    hp: str
    speed: str
    skills: str
    damage_resistances: str
    damage_immunities: str
    condition_immunities: str
    senses: str
    languages: str
    challenge_rating: str
    special_traits: List[NamedRulesPair]
    actions: List[NamedRulesPair]


@ai_model
class CombatStrategies(pydantic.BaseModel):
    """A list of combat strategies for a 5e monster."""

    strategies: List[NamedRulesPair]


@task(
    name="design-monster",
    cache_key_fn=ai_fn_task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=3,
)
@ai_fn
def design_monster(text: str) -> str:
    """
    Creatively design a Dungeons and Dragons fifth edition (5e) monster from the
    input text. Design monsters in the style of Bruce Cordell, Monte Cook,
    and Wolfgang Baur.

    Every Special Trait and Action should include 5e rules for how to use the trait
    or action.

    Return a stat block that contains the following:

    ```
    name: str
    ability_scores: str
    ac: str
    hp: str
    speed: str
    skills: str
    damage_resistances: str
    damage_immunities: str
    condition_immunities: str
    senses: str
    languages: str
    challenge_rating: str
    special_traits: List[NamedRulesPair]
    actions: List[NamedRulesPair]
    ```
    """


@task(
    name="design-combat-strategy",
    cache_key_fn=ai_fn_task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=3,
)
@ai_fn
def design_combat_strategy(monster: Monster) -> str:
    """
    Build a combat strategy for a Dungeons and Dragons fifth edition (5e)
    monster based on the input.

    The strategy should:
    - Reference specific actions, based on the monster's abilities, the monster will take
      before combat
    - Reference specific actions, based on the monster's abilities, the monster will take
      during combat
    - Reference specific actions, based on the monster's abilities, the monster will take
      after combat
    - Be appropriate to the creature's level of intelligence as defined by its intelligence
      ability score

    Actions the monster will take should reference the monster's abilities and
    should refer to the related 5e rules when appropriate.

    Return the strategies in the format List[NamedRulesPair].
    """


@task(
    name="design-adventure",
    cache_key_fn=ai_fn_task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=3,
)
@ai_fn
def design_adventure(monster: Monster) -> str:
    """
    Build an adventure for a Dungeons and Dragons fifth edition (5e) based on the
    input monster. The adventure should center on the monster and its lair.

    Design the adventure in the style of Bruce Cordell, Monte Cook, and Wolfgang
    Baur.

    Include the following:
    - A hook to get the players to the lair
    - A short description of the lair as a whole
    - A description of the monster's personality
    - A map of the lair using MermaidJS notation
    - A description of the monster's goals. List them.
    - A description of the monster's treasure. List it.
    - A description of the monster's minions. List them with explicit 5e rules.
    - A description of each room in the lair that appears in the MermaidJS map,
      what treasure is in the room using treasure from 5e, and what minions are in
      the room using monsters from 5e.
    
    Return the adventure as a Markdown document.
    """


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=3,
)
def parse_monster(monster_text: str) -> Monster:
    """Parse the monster from the text."""
    try:
        return Monster(monster_text)
    except pydantic.ValidationError as e:
        raise GenerationFailed(
            "The model failed to generate a suitable monster.", monster_text, e
        )


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=3,
    retry_delay_seconds=3,
)
def parse_strategy(strategy_text: str) -> CombatStrategies:
    """Parse the combat strategy from the text."""
    try:
        return CombatStrategies(strategy_text)
    except pydantic.ValidationError as e:
        raise GenerationFailed(
            "The model failed to generate suitable combat strategies.", strategy_text, e
        )


@flow
def generate_monster_combat_strategy(url: str) -> dict:
    text = get_text(get_page(url))
    monster_text = design_monster(text)
    monster = parse_monster(monster_text)
    strategy_text = design_combat_strategy(monster)
    adventure = design_adventure(monster)
    strategy = parse_strategy(strategy_text)

    create_markdown_artifact(
        key=slugify(monster.name),
        markdown=monster_template.render(**strategy.dict(), **monster.dict()),
        description=f"Stat block and combat strategy for {monster.name}",
    )

    create_markdown_artifact(
        key=f"{slugify(monster.name)}-adventure",
        markdown=adventure,
        description=f"Adventure for {monster.name}",
    )

    return {"monster": monster.dict(), **strategy.dict(), "adventure": adventure}


if __name__ == "__main__":
    generate_monster_combat_strategy(sys.argv[1])
