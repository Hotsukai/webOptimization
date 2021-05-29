# import sys
# sys.path.append('.')
from flask import Flask, render_template, request, Response
import io
from GPTSAgent import GPUCBAgent, getPlt

app = Flask(__name__)
agent = GPUCBAgent()
x, ucb = agent.get_arm()


@app.route("/test", methods=["GET"])
def ping_test():
    return "hello world"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        # FIXME 更新処理をする
        print(x, float(request.json["value"]))
        agent.sample(x, float(request.json["value"]))
        return {"status": "ok"}


@app.route("/image.png", methods=["GET"])
def plot():
    plt = getPlt(agent, x, ucb)
    output = io.BytesIO()
    plt.savefig(output, format="png")
    return Response(output.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
