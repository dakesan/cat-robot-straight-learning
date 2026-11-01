(async () => {
  const pause = ms => globalThis.__advance ? globalThis.__advance(ms) : new Promise(resolve => setTimeout(resolve, ms));
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const calls = () => fetch('/__calls').then(response => response.json());
  const byId = id => document.getElementById(id);
  await pause(600);
  assert(byId('arm').disabled === false, 'The page must start disarmed');
  assert((await calls()).filter(call => call.path === '/api/command').length === 0, 'No startup motion commands');
  byId('arm').click();
  await pause(200);
  assert(byId('arm').disabled, 'Arming must enable controls');
  const forward = document.querySelector('[data-left="1"][data-right="1"]');
  const buttonStyle = getComputedStyle(forward);
  assert(buttonStyle.userSelect === 'none' || buttonStyle.webkitUserSelect === 'none', 'Button labels must not be selectable');
  for (const type of ['selectstart', 'contextmenu', 'dragstart', 'touchstart']) {
    const event = new Event(type, {bubbles: true, cancelable: true});
    forward.dispatchEvent(event);
    assert(event.defaultPrevented, `${type} must not interrupt holding a button`);
  }
  forward.setPointerCapture = () => {};
  forward.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 11, bubbles: true}));
  await pause(1500);
  let last = (await calls()).filter(call => call.path === '/api/command').at(-1);
  assert(last.left === '20' && last.right === '20', 'Holding forward must send both motors');
  assert((await calls()).filter(call => call.path === '/api/command' && call.left === '20').length >= 10,
    'Long holding must keep sending movement commands');
  forward.dispatchEvent(new PointerEvent('pointerup', {pointerId: 11, bubbles: true}));
  await pause(180);
  last = (await calls()).filter(call => call.path === '/api/command').at(-1);
  assert(last.left === '0' && last.right === '0', 'Releasing forward must stop both motors');
  for (const [label, left, right] of [['左旋回', '20', '-20'], ['右旋回', '-20', '20'], ['後ろへ', '-20', '-20']]) {
    const button = [...document.querySelectorAll('button')].find(element => element.textContent === label);
    assert(button, `Missing button: ${label}`);
    button.setPointerCapture = () => {};
    button.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 12, bubbles: true}));
    await pause(180);
    last = (await calls()).filter(call => call.path === '/api/command').at(-1);
    assert(last.left === left && last.right === right, `Wrong motor command for ${label}`);
    button.dispatchEvent(new PointerEvent('pointerup', {pointerId: 12, bubbles: true}));
    await pause(180);
    last = (await calls()).filter(call => call.path === '/api/command').at(-1);
    assert(last.left === '0' && last.right === '0', `Releasing ${label} must stop both motors`);
  }
  byId('servoEnabled').checked = true;
  byId('servo').value = '75';
  await pause(180);
  last = (await calls()).filter(call => call.path === '/api/command').at(-1);
  assert(last.servo === '1575', 'Servo input must use the configured pulse range');
  byId('stop').click();
  await pause(200);
  assert((await calls()).at(-1).path === '/api/stop', 'Stop must send the disarm request');
  assert(!byId('servoEnabled').checked && !byId('arm').disabled, 'Stop must disable servo and permit explicit rearming');
  byId('invertLeft').checked = true;
  byId('invertLeft').dispatchEvent(new Event('change'));
  await pause(650);
  assert(byId('invertLeft').checked, 'Polling must preserve unsaved settings');
  byId('save').click();
  await pause(200);
  assert((await calls()).at(-1).invert_left === '1', 'Saving must send inversion configuration');
  byId('reverseHeading').click();
  await pause(200);
  last = (await calls()).at(-1);
  assert(last.invert_left === '0' && last.invert_right === '1', 'Changing heading must toggle both motors');
  return {passed: true, checks: ['startup disarmed', 'arm', 'selection/callout blocked', 'long hold/release', 'turn left/right', 'backward', 'servo', 'stop', 'settings', 'reverse heading'],
    viewport: innerWidth, overflow: document.documentElement.scrollWidth > innerWidth};
})()
