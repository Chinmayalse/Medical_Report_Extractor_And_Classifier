from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
from test_identification import process_pdf
import psycopg2
from psycopg2.extras import Json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_filename(filename):
    cleaned = secure_filename(filename)
    return cleaned.lower()

def save_to_database(filename, processed_data, report_type, patient_name=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # First, ensure we have the processed_data as a dictionary
        if isinstance(processed_data, str):
            try:
                processed_data = json.loads(processed_data)
            except json.JSONDecodeError:
                processed_data = {"raw_text": processed_data}

        # Extract classification reason
        classification_reason = ""
        if isinstance(processed_data, dict):
            # Try different possible locations of the reason
            if "classification_result" in processed_data:
                classification_reason = processed_data["classification_result"].get("reason", "")
            else:
                classification_reason = processed_data.get("reason", "")

        # Insert into report_results
        insert_query = """
        INSERT INTO report_results 
        (filename, processed_data, report_type, patient_name, created_at, status)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        cur.execute(insert_query, (
            filename,
            Json(processed_data),
            report_type,
            patient_name,
            datetime.now(),
            'completed'
        ))
        
        result_id = cur.fetchone()[0]
        
        # Insert into report_summary with the reason
        summary_query = """
        INSERT INTO report_summary 
        (report_id, filename, patient_name, report_type, reason, created_at)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        cur.execute(summary_query, (
            result_id,
            filename,
            patient_name,
            report_type,
            classification_reason,  # Use the extracted reason
            datetime.now()
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return result_id
    
    except Exception as e:
        print(f"Error saving to database: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        filename = clean_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = process_pdf(filepath)
        
        if result:
            try:
                # Ensure result is valid JSON
                if isinstance(result, str):
                    result_data = json.loads(result)
                else:
                    result_data = result
                
                report_type = result_data.get('test', 'Unknown')
                patient_name = result_data.get('Patient Name', None)
                
                # Extract reason from classification result if available
                # if 'classification_result' in result_data:
                #     reason = result_data['classification_result'].get('reason', '')
                # else:
                #     reason = result_data.get('reason', '')
                
                db_id = save_to_database(
                    filename=filename,
                    processed_data=result_data,
                    report_type=report_type,
                    patient_name=patient_name,
                )
                
                if db_id:
                    return jsonify({
                        'report_id': db_id,
                        'message': 'File uploaded and processed successfully',
                        'filename': filename
                    }), 200
                
                return jsonify({
                    'error': 'Failed to save to database',
                    'filename': filename
                }), 500
                
            except json.JSONDecodeError as e:
                return jsonify({
                    'error': f'Invalid JSON format: {str(e)}',
                    'filename': filename
                }), 500
            except Exception as e:
                return jsonify({
                    'error': f'Error processing file: {str(e)}',
                    'filename': filename
                }), 500
        
        return jsonify({
            'error': 'Failed to process file',
            'filename': filename
        }), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report/<int:report_id>', methods=['GET'])
def get_report_by_id(report_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT filename, processed_data, created_at, report_type, patient_name 
            FROM report_results 
            WHERE id = %s
        """, (report_id,))
        
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            processed_data = result[1]  # This is the JSONB data from the database
            
            # Structure the response
            report_data = {
                'report_type': result[3],
                'patient_name': result[4],
                'hospital_info': {
                    'name': processed_data.get('hospital_lab_name', ''),
                    'location': processed_data.get('hospital_lab_location', '')
                },
                'parameters': processed_data.get('parameters', {})
            }
            
            return jsonify(report_data), 200
        
        return jsonify({'error': 'Report not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Add a new route to get summary information
@app.route('/report-summary/<int:report_id>', methods=['GET'])
def get_report_summary(report_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT filename, patient_name, report_type, reason, created_at 
            FROM report_summary 
            WHERE report_id = %s
        """, (report_id,))
        
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            summary_data = {
                'filename': result[0],
                'patient_name': result[1],
                'report_type': result[2],
                'reason': result[3],
                'created_at': result[4].strftime('%Y-%m-%d %H:%M:%S')
            }
            return jsonify(summary_data), 200
        
        return jsonify({'error': 'Report summary not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)