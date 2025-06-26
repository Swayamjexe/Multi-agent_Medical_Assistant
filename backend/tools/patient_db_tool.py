import sqlite3
import logging
from typing import Dict, Optional, List

class PatientDBTool:
    def __init__(self, db_path: str = 'backend/data/patients.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    def get_patient_by_name(self, name: str) -> Optional[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patients WHERE patient_name LIKE ?', (f"%{name}%",))
            result = cursor.fetchone()
            conn.close()
            if result:
                columns = ['id', 'patient_name', 'discharge_date', 'primary_diagnosis',
                           'medications', 'dietary_restrictions', 'follow_up',
                           'warning_signs', 'discharge_instructions', 'created_at']
                return dict(zip(columns, result))
            return None
        except Exception as e:
            self.logger.error(f"Error looking up patient: {e}")
            return None

    def get_patients_by_name(self, name: str) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patients WHERE patient_name LIKE ?', (f"%{name}%",))
            results = cursor.fetchall()
            conn.close()
            columns = ['id', 'patient_name', 'discharge_date', 'primary_diagnosis',
                       'medications', 'dietary_restrictions', 'follow_up',
                       'warning_signs', 'discharge_instructions', 'created_at']
            return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            self.logger.error(f"Error looking up patients: {e}")
            return [] 