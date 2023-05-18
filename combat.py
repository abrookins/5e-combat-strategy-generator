import json
import os
import sys
from types import FunctionType
from typing import Any, Dict, List, Optional
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import timedelta

import pydantic
from jinja2 import Environment, FileSystemLoader
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.utilities.hashing import hash_objects
from prefect.context import TaskRunContext
from readability import Document
from marvin import ai_fn, ai_model


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
    """A variation of `task_input_hash` that doesn't hash the task function's code."""
    return hash_objects(
        context.task.task_key,
        arguments,
    )


@task(cache_key_fn=ai_fn_task_input_hash, cache_expiration=timedelta(days=1))
def get_page(url):
    """Get the HTML page from the URL."""
    return urlopen(url).read()


@task(cache_key_fn=ai_fn_task_input_hash, cache_expiration=timedelta(days=1))
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
)
@ai_fn
def design_combat_strategy(monster: Monster):
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


@flow
def generate_monster_combat_strategy(url: str) -> dict:
    text = get_text(get_page(url))
    monster_text = design_monster(text)

    try:
        monster = Monster(monster_text)
    except pydantic.ValidationError as e:
        raise GenerationFailed(
            "The model failed to generate a suitable monster.", monster_text, e
        )

    strategy_text = design_combat_strategy(monster)
    try:
        strategy = CombatStrategies(strategy_text)
    except pydantic.ValidationError as e:
        raise GenerationFailed(
            "The model failed to generate suitable combat strategies.", strategy_text, e
        )

    create_markdown_artifact(
        monster_template.render(**strategy.dict(), **monster.dict())
    )

    return {"monster": monster.dict(), **strategy.dict()}


if __name__ == "__main__":
    generate_monster_combat_strategy(sys.argv[1])
