from raglens import Answerer
from raglens.answer import NOT_FOUND
from raglens.llm import LLMError, OpenAICompatible


def test_extractive_answer_has_citations(retriever):
    ans = Answerer(retriever).ask("How much does the Team plan cost per seat?")
    assert ans.found and ans.mode == "extractive"
    assert "£24" in ans.text and "[1]" in ans.text
    assert ans.citations[0].doc == "billing.md" and ans.citations[0].heading == "Plans"


def test_unanswerable_question(retriever):
    ans = Answerer(retriever).ask("Do you sell bicycles on Jupiter?")
    assert ans.text == NOT_FOUND and not ans.found and not ans.citations


class FakeLLM(OpenAICompatible):
    def __init__(self, reply: str | Exception):
        super().__init__(api_key="test-key", base_url="http://localhost:0")
        self.reply = reply
        self.messages = None

    def chat(self, messages, temperature=0.1):
        self.messages = messages
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_llm_answer_parses_citations(retriever):
    llm = FakeLLM("Team costs £24 per seat [1], with SSO included [2]. Ignore [9].")
    ans = Answerer(retriever, llm=llm).ask("How much does the Team plan cost?")
    assert ans.mode == "llm"
    assert [c.number for c in ans.citations] == [1, 2]
    assert "Context:" in llm.messages[1]["content"]


def test_llm_failure_falls_back_offline(retriever):
    ans = Answerer(retriever, llm=FakeLLM(LLMError("boom"))).ask("What encryption is used at rest?")
    assert ans.mode == "extractive" and "AES-256" in ans.text


def test_llm_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not OpenAICompatible().available
