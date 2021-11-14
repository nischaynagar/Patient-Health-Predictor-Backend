const { spawn } = require('child_process');

// const childPython = spawn('python3', ['--version']);
// const childPython = spawn('python', ['main.py']);



args = ['prediction.py', 4, 30,1,0.7,0.2,194,32,36,7.5,3.6,0.92];


const childPython = spawn('python3', args);

childPython.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
});

childPython.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
});

childPython.on('close', (data) => {
    console.log(`child process exited with code ${data}`);
});