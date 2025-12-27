# orchestrator.py

from services.inference_service import InferenceService
from agents.validation.agent import ValidationAgent
# from agents.analysis.agent import AnalysisAgent
# from agents.mitigation.agent import MitigationAgent
# from agents.execution.agent import ExecutionAgent
# from policy.policy_engine import PolicyEngine

class Orchestrator:
    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        self.validation_agent = ValidationAgent()
        # self.analysis_agent = AnalysisAgent()
        # self.mitigation_agent = MitigationAgent()
        # self.execution_agent = ExecutionAgent()
        # self.policy_engine = PolicyEngine()

    def process_flow(self, flow):
        """
        Process a single network flow through the full agentic pipeline
        based on inference service output.
        """
        # Step 1: Run autoencoder inference
        is_anomaly, score, details = self.inference_service.predict(flow)
        
        if not is_anomaly:
            # Flow is normal → log / monitor only
            return {"status": "normal", "score": score}

        # Step 2: Route anomalous flow to Validation Agent
        validation_result = self.validation_agent.evaluate(flow, details)
        if validation_result["verdict"] != "confirmed_attack":
            return {"status": "false_positive", "score": score}

        # Step 3: Analysis Agent interprets attack
        analysis_result = self.analysis_agent.analyze(flow, validation_result)

        # Step 4: Mitigation planning
        mitigation_plan = self.mitigation_agent.plan(flow, analysis_result)

        # Step 5: Apply policy checks
        if self.policy_engine.is_allowed(mitigation_plan):
            execution_result = self.execution_agent.execute(mitigation_plan)
        else:
            execution_result = {"status": "blocked_by_policy"}

        # Step 6: Return full pipeline results
        return {
            "status": "handled",
            "score": score,
            "validation": validation_result,
            "analysis": analysis_result,
            "mitigation_plan": mitigation_plan,
            "execution": execution_result
        }
