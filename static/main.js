new Vue({
  el: "#app",
  data: {
    count: 0,
    imageSrcs: ["image.png"],
    value: 0,
  },
  created(){
    this.reset()
  },
  computed:{
    reversedImageSrcs(){
      return this.imageSrcs.reverse()
    }
  },
  methods: {
    submitSample() {
      const obj = { value: this.value };
      const method = "POST";
      const body = JSON.stringify(obj);
      const headers = {
        Accept: "application/json",
        "Content-Type": "application/json",
      };
      fetch("/", { method, headers, body })
        .then(() => {
          this.imageSrcs.push(`image.png?dummy=${this.count}`);
        })
        .catch(console.error);
      this.count++;
    },
    reset() {
      fetch("/reset")
        .then(() => {
          this.imageSrcs = ["image.png"];
        })
        .catch(console.error);
      this.count++;
    },
  },
});
