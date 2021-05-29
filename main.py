# import sys
# sys.path.append('.')
from flask import Flask, render_template, request, Response
import io
from Agents import GPUCBAgent, getPlt
import pickle

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    return "hello world"


@app.route("/", methods=["GET", "POST"])
def main_handler():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        with open("agent.dat", "rb") as f:
            agent = pickle.load(f)
        x, _ = agent.get_arm()
        print(x)
        agent.sample(x, float(request.json["value"]))
        with open("agent.dat", "wb") as f:
            pickle.dump(agent, f)
        return {"status": "ok"}


@app.route("/image.png", methods=["GET"])
def get_image():
    with open("agent.dat", "rb") as f:
        agent = pickle.load(f)
    x, ucb = agent.get_arm()
    plt = getPlt(agent, x, ucb)
    output = io.BytesIO()
    plt.savefig(output, format="png")
    return Response(output.getvalue(), mimetype="image/png")


def reset_agent():
    agent = GPUCBAgent()
    with open("agent.dat", "wb") as f:
        pickle.dump(agent, f)


@app.route("/reset", methods=["GET"])
def call_reset_agent():
    reset_agent()
    return {"status": "ok"}


if __name__ == "__main__":
    reset_agent()
    app.run(debug=True)
