import logging
from typing import Optional, Dict, List
from backend.tools.patient_db_tool import PatientDBTool

class ReceptionistAgent:
    def __init__(self, llm):
        self.llm = llm  # Should be ChatOllama (Mistral)
        self.db_tool = PatientDBTool()
        self.logger = logging.getLogger(__name__)
        self.multiple_matches: Optional[List[Dict]] = None
        self.patient: Optional[Dict] = None

    def greet(self) -> str:
        return "Hello! Welcome to the Nephrology Assistant. May I have your name to look up your records?"

    def extract_name(self, message: str) -> Optional[str]:
        import re
        patterns = [
            r"I'm\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"I am\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"My name is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"Call me\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"I'm\s+([A-Z][a-z]+)",
            r"I am\s+([A-Z][a-z]+)",
            r"My name is\s+([A-Z][a-z]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def lookup_patient_by_name(self, name: str) -> List[Dict]:
        # Return all matches for the name
        return self.db_tool.get_patients_by_name(name)

    def handle_name(self, name: str):
        matches = self.lookup_patient_by_name(name)
        if not matches:
            return f"❌ Sorry, I couldn't find any patient named '{name}'. Please check and try again."
        if len(matches) > 1:
            options = "\n".join([f"{i+1}. {m.get('patient_name', m.get('name',''))} (age: {m.get('age','?')})" for i, m in enumerate(matches)])
            self.multiple_matches = matches
            return f"🔍 I found multiple matches:\n{options}\nPlease type the number of your record."
        self.patient = matches[0]
        return self._summarize_patient()

    def select_match(self, index: int):
        if self.multiple_matches:
            try:
                self.patient = self.multiple_matches[index - 1]
                self.multiple_matches = None
                return self._summarize_patient()
            except IndexError:
                return "❌ Invalid number. Please try again."
        return "❌ No multiple options to choose from."

    def _summarize_patient(self):
        p = self.patient
        self.logger.info(f"Patient selected: {p.get('patient_name', p.get('name',''))} (ID: {p.get('id', p.get('patient_id',''))})")
        summary = (
            f"📄 Found your discharge summary:\n"
            f"Name: {p.get('patient_name', p.get('name',''))}\n"
            f"Age: {p.get('age','?')}\n"
            f"Diagnosis: {p.get('primary_diagnosis', p.get('diagnosis',''))}\n"
            f"Medications: {', '.join(p.get('medications', [])) if isinstance(p.get('medications'), list) else p.get('medications','')}\n"
            f"Follow-up: {p.get('follow_up','')}\n"
            f"Instructions: {p.get('discharge_instructions', ', '.join(p.get('instructions', [])))}\n\n"
            f"How have you been feeling since your discharge?\n"
            f"(You can also ask any medical questions, and I'll bring in our clinical expert.)"
        )
        return summary

    def is_medical_query(self, message: str) -> bool:
        # Always route to clinical agent for strong nephrology keywords
        strong_keywords = ["kidney", "ckd", "nephropathy", "glomerular", "proteinuria", "creatinine", "nephrotoxic", "bp", "pressure", "swelling"]
        if any(term in message.lower() for term in strong_keywords):
            self.logger.info(f"[DEBUG] Strong nephrology keyword detected in message: {message}")
            return True
        prompt = (
            "Determine if the following user message is a medical or health-related question. "
            "Reply with only 'yes' or 'no'.\n\n"
            f"Message: {message}\n\nAnswer:"
        )
        try:
            result = self.llm(prompt)[0]["generated_text"].strip().lower()
            self.logger.info(f"[DEBUG] LLM medical classification result: {result}")
            return "yes" in result[:10]  # more lenient check
        except Exception:
            self.logger.warning("LLM failed to classify message, falling back to keyword match.")
            return any(k in message.lower() for k in [
                "pain", "symptom", "medication", "treatment", "diagnosis", "doctor", "nurse",
                "hospital", "clinic", "disease", "condition", "swelling", "bp", "pressure", "nephropathy"
            ])

    def handle_message(self, message: str) -> Dict:
        if self.is_medical_query(message):
            return {"route_to": "clinical", "response": "I will connect you to our clinical agent for your medical question."}
        else:
            return {"route_to": "receptionist", "response": "How else may I assist you?"} 