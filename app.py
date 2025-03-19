from flask import Flask, request, jsonify
from flask_cors import CORS  
import tax_assistance  

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/result', methods=["POST"])
def result():
    try:
        data = request.json
        question = data.get("question")
        category = data.get("incomecategory")

        if not question or not category:  # Ensure required fields are present
            return jsonify({"error": "Missing required fields"}), 400

        # Ensure category is a valid input before calling get_vector_data
        if not isinstance(category, str):
            return jsonify({"error": "Invalid category type"}), 400

        result1 = tax_assistance.get_vector_data(question, category)  # Call function safely
        return jsonify({"tax": result1})

    except IndexError as e:
        return jsonify({"error": "List index out of range. Check input values."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/caculatededuction', methods=["POST"])
def caculatededuction():
    try:
        data = request.json
        finalquestion = data.get("finalQuestion")
        deductiondoc=data.get("deductiondoc")
        
        answer=tax_assistance.get_answer(finalquestion, deductiondoc)
        return jsonify({"answer": answer,"taxableincome":tax_assistance.extract_taxable_income(answer)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

if __name__ == "__main__":
    app.run(debug=True, port=8000)
