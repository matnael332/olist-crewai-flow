"""Crew 1: the Data Analyst crew (6 agents).

Tool-bearing agents run the deterministic tools (results returned verbatim); tool-less narrator
agents write the structured narratives (output_pydantic). Gemini only returns reliable structured
output from agents without tools, so the two roles are separate agents.
"""

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from pulse_core.narratives import CleaningNarrative, ContractNarrative, Insights
from pulse_flow.llm import make_llm
from pulse_flow.runctx import get_context
from pulse_flow.tools.analyst_tools import draft_contract_schema, ingest_and_clean, run_eda

MAX_RPM = 5


@CrewBase
class DataAnalystCrew:
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

    @agent
    def data_engineer(self) -> Agent:
        return self._agent("data_engineer", [ingest_and_clean])

    @agent
    def cleaning_documenter(self) -> Agent:
        return self._agent("cleaning_documenter")

    @agent
    def eda_analyst(self) -> Agent:
        return self._agent("eda_analyst", [run_eda])

    @agent
    def insights_writer(self) -> Agent:
        return self._agent("insights_writer")

    @agent
    def data_steward(self) -> Agent:
        return self._agent("data_steward", [draft_contract_schema])

    @agent
    def contract_author(self) -> Agent:
        return self._agent("contract_author")

    @task
    def ingest_clean_task(self) -> Task:
        return Task(config=self.tasks_config["ingest_clean_task"])

    @task
    def cleaning_narrative_task(self) -> Task:
        return Task(
            config=self.tasks_config["cleaning_narrative_task"], output_pydantic=CleaningNarrative
        )

    @task
    def eda_task(self) -> Task:
        return Task(config=self.tasks_config["eda_task"])

    @task
    def insights_task(self) -> Task:
        return Task(config=self.tasks_config["insights_task"], output_pydantic=Insights)

    @task
    def contract_task(self) -> Task:
        return Task(config=self.tasks_config["contract_task"])

    @task
    def contract_narrative_task(self) -> Task:
        return Task(
            config=self.tasks_config["contract_narrative_task"], output_pydantic=ContractNarrative
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="DataAnalystCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            planning=False,
            cache=False,
            tracing=False,
            max_rpm=MAX_RPM,
            output_log_file=get_context().crew_log("data_analyst_crew"),
        )
