from triton_sdk.chat import ChatMessage, Role, SimpleChatTemplate


def test_simple_template_appends_generation_prompt() -> None:
    prompt = SimpleChatTemplate().render([ChatMessage(role=Role.USER, content="hi")])
    assert prompt == "user: hi\nassistant:"
