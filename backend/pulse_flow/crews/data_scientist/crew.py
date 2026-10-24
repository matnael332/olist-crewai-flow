"""Crew 2: the Data Scientist crew (5 agents, one class, two phases).

The Flow runs feature_phase(), then its own required-features gate, then modeling_phase(), so the
"required features exist before modeling" check sits at Flow level between the phases. crew() is
the canonical full crew (all agents and tasks), used for documentation and tests.
"""

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, task
from crewai.project import crew as crew_definition  # a method named crew would shadow it

from pulse_core.narratives import EvaluationNarrative, FeatureNarrative, ModelCardNarrative
from pulse_flow.llm import make_llm
from pulse_flow.runctx import get_context
from pulse_flow.tools.scientist_tools import build_features, train_models, validate_inputs

MAX_RPM = 5
CREW_NAME = "DataScientistCrew"


@CrewBase
class DataScientistCrew:
    agents: list[BaseAgent]
    tasks: list[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _agent(self, key: str, tools: list | None = None) -> Agent:
        return Agent(
            config=self.agents_config[key],
            llm=make_llm(get_context().model),
            tools=tools or [],
            max_iter=6,
            max_retry_limit=1,
            allow_delegation=False,
            cache=False,
            verbose=True,
        )

    def _crew(self, agents: list, tasks: list) -> Crew:
        return Crew(
            name=CREW_NAME,
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            planning=False,
            cache=False,
            tracing=False,
            max_rpm=MAX_RPM,
            output_log_file=get_context().crew_log("data_scientist_crew"),
        )

    @agent
    def feature_engineer(self) -> Agent:
        return self._agent("feature_engineer", [validate_inputs, build_features])

    @agent
    def feature_reviewer(self) -> Agent:
        return self._agent("feature_reviewer")

    @agent
    def model_trainer(self) -> Agent:
        return self._agent("model_trainer", [train_models])

    @agent
    def model_evaluator(self) -> Agent:
        return self._agent("model_evaluator")

    @agent
    def model_card_author(self) -> Agent:
        return self._agent("model_card_author")

    @task
    def feature_task(self) -> Task:
        return Task(config=self.tasks_config["feature_task"])

    @task
    def feature_narrative_task(self) -> Task:
        return Task(
            config=self.tasks_config["feature_narrative_task"], output_pydantic=FeatureNarrative
        )

    @task
    def training_task(self) -> Task:
        return Task(config=self.tasks_config["training_task"])

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluation_task"], output_pydantic=EvaluationNarrative
        )

    @task
    def model_card_task(self) -> Task:
        return Task(config=self.tasks_config["model_card_task"], output_pydantic=ModelCardNarrative)

    @crew_definition
    def crew(self) -> Crew:
        return self._crew(self.agents, self.tasks)

    @crew_definition
    def feature_phase(self) -> Crew:
        return self._crew(
            [self.feature_engineer(), self.feature_reviewer()],
            [self.feature_task(), self.feature_narrative_task()],
        )

    @crew_definition
    def modeling_phase(self) -> Crew:
        return self._crew(
            [self.model_trainer(), self.model_evaluator(), self.model_card_author()],
            [self.training_task(), self.evaluation_task(), self.model_card_task()],
        )
