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

    # Check for top-level keys and specific values
    assert "version" in raw_config
    assert isinstance(raw_config["version"], str)

    assert raw_config.get("hertz") == 1
    assert isinstance(raw_config["hertz"], (int, float))

    assert raw_config.get("name") == "spot_speak"
    assert isinstance(raw_config["name"], str)

    assert "api_key" in raw_config
    assert isinstance(raw_config["api_key"], str)

    assert "system_prompt_base" in raw_config
    assert isinstance(raw_config["system_prompt_base"], str)

    assert "system_governance" in raw_config
    assert isinstance(raw_config["system_governance"], str)

    assert "system_prompt_examples" in raw_config
    assert isinstance(raw_config["system_prompt_examples"], str)

    # Validate agent_inputs
    agent_inputs = raw_config.get("agent_inputs", [])
    assert isinstance(agent_inputs, list)
    assert len(agent_inputs) > 0, "agent_inputs should not be empty"
    assert agent_inputs[0]["type"] == "GoogleASRInput"
    for input_config in agent_inputs:
        assert_input_class_exists(input_config)

    # Validate cortex_llm
    cortex_llm = raw_config.get("cortex_llm", {})
    assert isinstance(cortex_llm, dict)
    assert cortex_llm.get("type") == "OpenAILLM", f"'type' key in cortex_llm is not OpenAILLM"
    assert get_llm_class(cortex_llm["type"]) is not None

    # Validate simulators
    simulators = raw_config.get("simulators", [])
    assert isinstance(simulators, list)
    assert len(simulators) > 0, "simulators should not be empty"
    assert simulators[0]["type"] == "WebSim"
    for simulator in simulators:
        assert get_simulator_class(simulator["type"]) is not None

    # Validate agent_actions
    agent_actions = raw_config.get("agent_actions", [])
    assert isinstance(agent_actions, list)
    assert len(agent_actions) > 0, "agent_actions should not be empty"
    speak_action = agent_actions[0]
    assert speak_action["name"] == "speak"
    assert speak_action["llm_label"] == "speak"
    assert speak_action["implementation"] == "passthrough"
    assert speak_action["connector"] == "elevenlabs_tts"
    for action in agent_actions:
        assert_action_classes_exist(action)
