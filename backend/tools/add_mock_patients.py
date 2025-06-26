import sqlite3
import json

DB_PATH = 'backend/data/patients.db'
MOCK_FILE = 'data/mock_patients.txt'

def add_mock_patients():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    with open(MOCK_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            patient = json.loads(line)
            # Convert medications to string if it's a list
            if isinstance(patient.get('medications'), list):
                patient['medications'] = ', '.join(patient['medications'])
            cursor.execute('''
                INSERT INTO patients (
                    patient_name, discharge_date, primary_diagnosis, medications,
                    dietary_restrictions, follow_up, warning_signs, discharge_instructions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient.get('patient_name'),
                patient.get('discharge_date'),
                patient.get('primary_diagnosis'),
                patient.get('medications'),
                patient.get('dietary_restrictions'),
                patient.get('follow_up'),
                patient.get('warning_signs'),
                patient.get('discharge_instructions')
            ))
    conn.commit()
    conn.close()
    print('Mock patients added successfully.')

if __name__ == '__main__':
    add_mock_patients() 