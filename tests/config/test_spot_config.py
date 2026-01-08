import importlib
import os
from typing import Optional, Type

import json5

from actions.base import ActionConnector, Interface
from inputs import find_module_with_class
from inputs.base import Sensor
from llm import get_llm_class
from simulators import get_simulator_class


def find_subclass_in_module(module, parent_class: Type) -> Optional[Type]:
    """Find a subclass of parent_class in the given module."""
    for _, obj in module.__dict__.items():
        if (
            isinstance(obj, type)
            and issubclass(obj, parent_class)
            and obj != parent_class
        ):
            return obj
    return None


def assert_input_class_exists(input_config):
    """Assert that the input class exists without instantiating it."""
    class_name = input_config["type"]
    module_name = find_module_with_class(class_name)
    assert module_name is not None, f"Input class '{class_name}' not found"

    module = importlib.import_module(f"inputs.plugins.{module_name}")
    input_class = find_subclass_in_module(module, Sensor)
    assert input_class is not None, f"No Sensor subclass found for '{class_name}'"


def assert_action_classes_exist(action_config):
    """Assert that all required classes for an action exist without instantiating them."""
    # Check interface exists
    action_module = importlib.import_module(
        f"actions.{action_config['name']}.interface"
    )
    interface = find_subclass_in_module(action_module, Interface)
    assert (
        interface is not None
    ), f"No interface found for action {action_config['name']}"

    # Check connector exists
    try:
        connector_module = importlib.import_module(
            f"actions.{action_config['name']}.connector.{action_config['connector']}"
        )
        connector = find_subclass_in_module(connector_module, ActionConnector)
        assert (
            connector is not None
        ), f"No connector found for action {action_config['name']}"
    except (ImportError, ModuleNotFoundError):
        assert False, f"Connector module not found for action {action_config['name']}"


def test_spot_config():
    """Test that the spot.json5 config file can be loaded and is valid."""
    config_path = os.path.join(os.path.dirname(__file__), "../../config/spot.json5")
    
    with open(config_path, "r") as f:
        raw_config = json5.load(f)

    # Check for top-level keys
    assert "version" in raw_config
    assert isinstance(raw_config["version"], str)

    assert "hertz" in raw_config
    assert isinstance(raw_config["hertz"], (int, float))

    assert "name" in raw_config
    assert isinstance(raw_config["name"], str)

    assert "api_key" in raw_config
    assert isinstance(raw_config["api_key"], str)

    assert "system_prompt_base" in raw_config
    assert isinstance(raw_config["system_prompt_base"], str)

    assert "system_governance" in raw_config
    assert isinstance(raw_config["system_governance"], str)

    assert "system_prompt_examples" in raw_config
    assert isinstance(raw_config["system_prompt_examples"], str)

    agent_inputs = raw_config.get("agent_inputs", [])
    assert isinstance(agent_inputs, list)

    cortex_llm = raw_config.get("cortex_llm", {})
    assert isinstance(cortex_llm, dict)
    assert "type" in cortex_llm, f"'type' key missing in cortex_llm of spot.json5"
    assert get_llm_class(cortex_llm["type"]) is not None

    simulators = raw_config.get("simulators", [])
    assert isinstance(simulators, list)

    agent_actions = raw_config.get("agent_actions", [])
    assert isinstance(agent_actions, list)

    for input_config in agent_inputs:
        assert_input_class_exists(input_config)

    for simulator in simulators:
        assert get_simulator_class(simulator["type"]) is not None

    for action in agent_actions:
        assert_action_classes_exist(action)
