import * as tf from "@tensorflow/tfjs";

import plop from "./plop.json";

const generate_text = (model, start_string) => {
  return new Promise(resolve => {
    const num_generate = 300 - start_string.length;

    // input_eval = [char2idx[s] for s in start_string]
    let input_eval = start_string
      .split("")
      .map(letter => plop[letter])
      .filter(id => id !== undefined);

    input_eval = tf.expandDims(
      input_eval.length !== 0
        ? input_eval
        : "Halla-aho"
            .split("")
            .map(letter => plop[letter])
            .filter(id => id !== undefined),
      0
    );

    let text_generated = "";

    const temperature = 0.8;

    model.resetStates();

    const gg = input_eval.arraySync();

    const flip = gg[0].reduce(
      (prev, id) => prev.concat(getKeyByValue(plop, id)),
      ""
    );

    for (let i = 0; i < num_generate; i++) {
      let predictions = model.predict(input_eval);

      predictions = tf.squeeze(predictions, 0);

      predictions = tf.div(predictions, tf.scalar(temperature));

      const naks = tf.multinomial(predictions, 1).arraySync();

      const predicted_id = naks.pop()[0];

      input_eval = tf.expandDims(predicted_id, 0);

      text_generated += getKeyByValue(plop, predicted_id);
    }

    resolve(flip + text_generated);
  });
};

tf.loadLayersModel("/model.json").then(model => {
  const intro = document.getElementById("intro");

  intro.innerHTML = "Kysy persunaattorilta:";

  const myText = document.getElementById("myText");

  myText.hidden = false;

  const myButton = document.getElementById("myButton");

  myButton.hidden = false;

  const myWisdoms = document.getElementById("myWisdoms");

  myForm.addEventListener("submit", e => {
    e.preventDefault();
    myText.hidden = true;
    myButton.hidden = true;
    const text = myText.value;
    const words = text.split(" ");
    const word = words[Math.floor(Math.random() * words.length)];
    myText.value = "";
    const youTitle = document.createElement("h4");
    youTitle.innerHTML = "YOU:";
    myWisdoms.appendChild(youTitle);
    const youText = document.createElement("p");
    youText.innerHTML = text;
    myWisdoms.appendChild(youText);
    const botTitle = document.createElement("h4");
    botTitle.innerHTML = "PERSUNAATTORI:";
    myWisdoms.appendChild(botTitle);
    const botText = document.createElement("p");
    botText.innerHTML = "Ladataan...";
    myWisdoms.appendChild(botText);
    setTimeout(() => {
      generate_text(model, word).then(wisdom => {
        botText.innerHTML = wisdom;
        myButton.hidden = false;
        myText.hidden = false;
      });
    }, 100);
  });
});

function getKeyByValue(object, value) {
  return Object.keys(object).find(key => object[key] === value);
}
