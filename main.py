# import sys
# sys.path.append('.')
from flask import Flask, render_template, request, Response
import io
from GPTSAgent import GPUCBAgent, getPlt
import pickle

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def ping_test():
    return "hello world"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        with open('agent.dat', 'rb') as f:
            agent = pickle.load(f)
        x, _ = agent.get_arm()
        print(x)
        agent.sample(x, float(request.json["value"]))
        with open('agent.dat', 'wb') as f:
            pickle.dump(agent, f)
        return {"status": "ok"}


@app.route("/image.png", methods=["GET"])
def plot():
    with open('agent.dat', 'rb') as f:
        agent = pickle.load(f)
    x, ucb = agent.get_arm()
    plt = getPlt(agent, x, ucb)
    output = io.BytesIO()
    plt.savefig(output, format="png")
    return Response(output.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    agent = GPUCBAgent()
    with open('agent.dat', 'wb') as f:
        pickle.dump(agent, f)
    app.run(debug=True)
