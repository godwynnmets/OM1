from enum import Enum
from typing import NamedTuple

from src.llm.function_schemas import (
    generate_function_schema_from_action,
    generate_function_schemas_from_actions,
    convert_function_calls_to_actions,
)
from src.llm.output_model import Action

# Mocking the necessary parts for the tests

class MockActionInput(NamedTuple):
    param_str: str
    param_int: int
    param_float: float
    param_bool: bool

class MockActionInterface:
    """This is a mock action."""
    input = MockActionInput

class MockAgentAction:
    def __init__(self, llm_label, interface, exclude_from_prompt=False):
        self.llm_label = llm_label
        self.interface = interface
        self.exclude_from_prompt = exclude_from_prompt

def test_generate_function_schema_from_action():
    mock_action = MockAgentAction(llm_label="mock_action", interface=MockActionInterface)

    schema = generate_function_schema_from_action(mock_action)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "mock_action"
    assert schema["function"]["description"] == "This is a mock action."
    
    properties = schema["function"]["parameters"]["properties"]
    assert "param_str" in properties
    assert properties["param_str"]["type"] == "string"
    assert "param_int" in properties
    assert properties["param_int"]["type"] == "integer"
    assert "param_float" in properties
    assert properties["param_float"]["type"] == "number"
    assert "param_bool" in properties
    assert properties["param_bool"]["type"] == "boolean"

    required = schema["function"]["parameters"]["required"]
    assert "param_str" in required
    assert "param_int" in required
    assert "param_float" in required
    assert "param_bool" in required

def test_generate_function_schemas_from_actions():
    mock_action_1 = MockAgentAction(llm_label="mock_action_1", interface=MockActionInterface)
    mock_action_2 = MockAgentAction(llm_label="mock_action_2", interface=MockActionInterface)
    mock_action_3_excluded = MockAgentAction(llm_label="mock_action_3", interface=MockActionInterface, exclude_from_prompt=True)

    actions = [mock_action_1, mock_action_2, mock_action_3_excluded]
    schemas = generate_function_schemas_from_actions(actions)

    assert len(schemas) == 2
    assert schemas[0]["function"]["name"] == "mock_action_1"
    assert schemas[1]["function"]["name"] == "mock_action_2"

def test_convert_function_calls_to_actions():
    function_calls = [
        {
            "function": {
                "name": "action_1",
                "arguments": '{"action": "do_something"}'
            }
        },
        {
            "function": {
                "name": "action_2",
                "arguments": '{"text": "hello world"}'
            }
        },
        {
            "function": {
                "name": "action_3",
                "arguments": '{"value": 42}'
            }
        },
        {
            "function": {
                "name": "action_4",
                "arguments": '{"command": "run"}'
            }
        },
        {
            "function": {
                "name": "action_5",
                "arguments": '{"other_param": "some_value"}'
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 5
    assert actions[0] == Action(type="action_1", value="do_something")
    assert actions[1] == Action(type="action_2", value="hello world")
    assert actions[2] == Action(type="action_3", value="42")
    assert actions[3] == Action(type="action_4", value="run")
    assert actions[4] == Action(type="action_5", value="some_value")

def test_convert_function_calls_to_actions_json_args():
    function_calls = [
        {
            "function": {
                "name": "action_1",
                "arguments": {"action": "do_something"}
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)
    assert len(actions) == 1
    assert actions[0] == Action(type="action_1", value="do_something")


class MockEnum(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"

class MockEnumActionInput(NamedTuple):
    param_enum: MockEnum

class MockEnumActionInterface:
    """This is a mock enum action."""
    input = MockEnumActionInput

def test_generate_function_schema_with_enum():
    mock_action = MockAgentAction(llm_label="mock_enum_action", interface=MockEnumActionInterface)

    schema = generate_function_schema_from_action(mock_action)
    
    properties = schema["function"]["parameters"]["properties"]
    assert "param_enum" in properties
    assert properties["param_enum"]["type"] == "string"
    assert properties["param_enum"]["enum"] == ["value1", "value2"]