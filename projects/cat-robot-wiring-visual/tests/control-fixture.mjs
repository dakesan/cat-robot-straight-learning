export function createFixture() {
  const calls = [];
  const state = {firmware: 'cat-robot/preview', armed: false, epoch: 0, reason: 'startup',
    left: 0, right: 0, encoder_left: 0, encoder_right: 0, invert_left: false,
    invert_right: false, battery_enabled: false, battery_mv: null};
  const response = (value, status = 200) => {
    const body = JSON.stringify(value);
    return {ok: status < 400, status, json: async () => JSON.parse(body)};
  };
  return async function fetchFixture(path, options = {}) {
    if (path === '/__calls') return response(calls);
    if (options.method === 'POST') {
      if (options.headers['X-Robot-Token'] !== 'preview-only') return response({}, 403);
      const fields = Object.fromEntries(new URLSearchParams(options.body));
      calls.push({path, ...fields});
      if (path === '/api/arm') {state.armed = true; state.epoch++;}
      if (path === '/api/stop') {state.armed = false; state.left = state.right = 0; state.reason = 'operator';}
      if (path === '/api/command') {state.left = +fields.left; state.right = +fields.right;}
      if (path === '/api/config') for (const key of ['invert_left','invert_right','battery_enabled']) state[key] = fields[key] === '1';
    }
    return response(state);
  };
}
