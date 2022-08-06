var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
canvas.width = 280;
canvas.height = 280;

var Mouse = { x: 0, y: 0 };
var lastMouse = { x: 0, y: 0 };
context.fillStyle = 'black';
context.fillRect(0, 0, canvas.width, canvas.height);
context.color = 'white';
context.lineWidth = 18;
context.lineJoin = context.lineCap = 'round';

canvas.addEventListener(
  'mousemove',
  function (e) {
    lastMouse.x = Mouse.x;
    lastMouse.y = Mouse.y;

    Mouse.x = e.pageX - this.offsetLeft - 15;
    Mouse.y = e.pageY - this.offsetTop - 15;
  },
  false
);

canvas.addEventListener(
  'mousedown',
  function (e) {
    canvas.addEventListener('mousemove', onPaint, false);
  },
  false
);

canvas.addEventListener(
  'mouseup',
  function () {
    canvas.removeEventListener('mousemove', onPaint, false);
  },
  false
);

var onPaint = function () {
  context.lineWidth = context.lineWidth;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.strokeStyle = context.color;

  context.beginPath();
  context.moveTo(lastMouse.x, lastMouse.y);
  context.lineTo(Mouse.x, Mouse.y);
  context.closePath();
  context.stroke();
};

function reset() {
  context.clearRect(0, 0, 280, 280);
  context.fillStyle = 'black';
  context.fillRect(0, 0, canvas.width, canvas.height);
  document.querySelector('#result').innerText = '';
}

function randomImage() {
  fetch('/randomkuva')
    .then((response) => response.json())
    .then((data) => {
      const mycanvas = document.getElementById('canvas');
      const mycontext = mycanvas.getContext('2d');
      const img = new Image();
      img.addEventListener(
        'load',
        () => {
          mycontext.drawImage(img, 0, 0, 28, 28, 0, 0, 280, 280);
        },
        false
      );
      img.src = 'data:image/png;base64,' + data.kuva;
      document.getElementById('result').innerText =
        'Tunnistettu numero: ' +
        data.tunnistus +
        ', oikea numero: ' +
        data.arvo;
    });
}

function predict() {
  const data = canvas.toDataURL();
  const response = fetch('/predict', { method: 'POST', body: data })
    .then((response) => response.json())
    .then((json) => {
      document.getElementById('result').innerText =
        'Tunnistettu numero: ' + json.numero;
      const img = new Image();
      img.addEventListener(
        'load',
        () => {
          context.drawImage(img, 0, 0, 28, 28, 0, 0, 280, 280);
        },
        false
      );
      img.src = 'data:image/png;base64,' + json.kuva;
    });
}
