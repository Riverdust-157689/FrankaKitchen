/* Minimal WebXR transport retained from XLeVR; no robot or task logic lives here. */
(function () {
  const state = { socket: null, leftGrip: false, rightGrip: false,
    leftTrigger: false, rightTrigger: false };

  function buttonMap(hand) {
    const tracked = hand.components && hand.components['tracked-controls'];
    const gamepad = tracked && tracked.controller && tracked.controller.gamepad;
    if (!gamepad) return {};
    return {
      a: !!(gamepad.buttons[4] && gamepad.buttons[4].pressed),
      b: !!(gamepad.buttons[5] && gamepad.buttons[5].pressed),
      squeeze: !!(gamepad.buttons[1] && gamepad.buttons[1].pressed),
      thumbstick: !!(gamepad.buttons[3] && gamepad.buttons[3].pressed)
    };
  }

  function sample(hand, side) {
    const object = hand.object3D;
    return {
      hand: side,
      position: {x: object.position.x, y: object.position.y, z: object.position.z},
      quaternion: {x: object.quaternion.x, y: object.quaternion.y,
        z: object.quaternion.z, w: object.quaternion.w},
      gripActive: side === 'right' ? state.rightGrip : state.leftGrip,
      trigger: (side === 'right' ? state.rightTrigger : state.leftTrigger) ? 1 : 0,
      buttons: buttonMap(hand)
    };
  }

  AFRAME.registerComponent('franka-teleop', {
    init: function () {
      this.left = document.querySelector('#leftHand');
      this.right = document.querySelector('#rightHand');
      const status = document.querySelector('#status');
      const sphere = document.querySelector('#statusSphere');
      const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
      const host = location.hostname || 'localhost';
      const wsPort = new URLSearchParams(location.search).get('ws_port') || '8001';
      state.socket = new WebSocket(`${scheme}://${host}:${wsPort}`);
      state.socket.onopen = () => { status.textContent = 'WebSocket: connected';
        status.style.color = '#4ade80'; sphere.setAttribute('color', '#22c55e'); };
      state.socket.onclose = () => { status.textContent = 'WebSocket: disconnected';
        status.style.color = '#f87171'; sphere.setAttribute('color', '#ef4444'); };
      state.socket.onerror = () => { status.textContent = 'WebSocket: error';
        status.style.color = '#f87171'; sphere.setAttribute('color', '#ef4444'); };
      this.bind(this.left, 'left');
      this.bind(this.right, 'right');
      this.lastSend = 0;
    },
    bind: function (hand, side) {
      hand.addEventListener('gripdown', () => state[side + 'Grip'] = true);
      hand.addEventListener('gripup', () => state[side + 'Grip'] = false);
      hand.addEventListener('triggerdown', () => state[side + 'Trigger'] = true);
      hand.addEventListener('triggerup', () => state[side + 'Trigger'] = false);
    },
    tick: function (time) {
      if (!state.socket || state.socket.readyState !== WebSocket.OPEN || time - this.lastSend < 16) return;
      this.lastSend = time;
      state.socket.send(JSON.stringify({timestamp: Date.now(),
        leftController: sample(this.left, 'left'), rightController: sample(this.right, 'right')}));
    }
  });

  document.addEventListener('DOMContentLoaded', function () {
    const scene = document.querySelector('a-scene');
    const install = () => scene.setAttribute('franka-teleop', '');
    scene.hasLoaded ? install() : scene.addEventListener('loaded', install);
    document.querySelector('#enter-vr').addEventListener('click', () => scene.enterVR(true));
    scene.addEventListener('enter-vr', () => document.querySelector('#panel').style.display = 'none');
    scene.addEventListener('exit-vr', () => document.querySelector('#panel').style.display = 'block');
  });
})();
