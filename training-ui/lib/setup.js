const { execFileSync, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.join(__dirname, '..', '..');

function execWindowsPowerShellSync(script, options = {}) {
    return execFileSync('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script], {
        ...options,
        stdio: options.stdio ?? 'pipe',
        windowsHide: options.windowsHide ?? true
    });
}

function getPkgGitHash(pkgDir) {
    try {
        const hash = execSync(`git log -1 --format=%H -- "${pkgDir}"`, {
            cwd: ROOT_DIR,
            stdio: 'pipe',
        }).toString().trim();
        return hash || null;
    } catch {
        return null;
    }
}

function pkgNeedsInstall(pkgDir) {
    const stampFile = path.join(pkgDir, '.installed_git_hash');
    const currentHash = getPkgGitHash(pkgDir);
    if (!fs.existsSync(stampFile)) return true;
    if (!currentHash) return false;
    const installedHash = fs.readFileSync(stampFile, 'utf8').trim();
    return installedHash !== currentHash;
}

function writePkgInstallStamp(pkgDir) {
    const hash = getPkgGitHash(pkgDir);
    if (hash) fs.writeFileSync(path.join(pkgDir, '.installed_git_hash'), hash, 'utf8');
}

function ensureCudaDirectBackend() {
    if (process.platform !== 'win32') return;
    const pkgPath = path.join(ROOT_DIR, 'cuda_direct_pkg');
    if (!fs.existsSync(pkgPath)) return;
    if (!pkgNeedsInstall(pkgPath)) return;

    const venvPath = path.join(ROOT_DIR, 'venv');
    const pythonPath = path.join(venvPath, 'Scripts', 'python.exe');
    const pythonCmd = fs.existsSync(pythonPath) ? `"${pythonPath}"` : 'python';

    console.log('[setup] Installing cuda_direct_backend...');
    try {
        execWindowsPowerShellSync(`${pythonCmd} -m pip install --no-deps -e "${pkgPath}"`, { stdio: 'pipe' });
        writePkgInstallStamp(pkgPath);
        console.log('[setup] cuda_direct_backend installed.\n');
    } catch {
        console.warn('[setup] Could not install cuda_direct_backend. Multi-GPU cuda_direct will be unavailable.\n');
    }
}

function ensureWdParallel() {
    const pkgPath = path.join(ROOT_DIR, 'wd_parallel_pkg');
    if (!fs.existsSync(pkgPath)) return;
    if (!pkgNeedsInstall(pkgPath)) return;

    const venvPath = path.join(ROOT_DIR, 'venv');
    let pythonCmd;
    if (process.platform === 'win32') {
        const pythonPath = path.join(venvPath, 'Scripts', 'python.exe');
        pythonCmd = fs.existsSync(pythonPath) ? `"${pythonPath}"` : 'python';
    } else {
        const pythonPath = path.join(venvPath, 'bin', 'python');
        pythonCmd = fs.existsSync(pythonPath) ? pythonPath : 'python3';
    }

    console.log('[setup] Installing wd_parallel...');
    try {
        if (process.platform === 'win32') {
            execWindowsPowerShellSync(`${pythonCmd} -m pip install --no-deps -e "${pkgPath}"`, { stdio: 'pipe' });
        } else {
            execSync(`${pythonCmd} -m pip install --no-deps -e '${pkgPath}'`, { stdio: 'pipe' });
        }
        writePkgInstallStamp(pkgPath);
        console.log('[setup] wd_parallel installed.\n');
    } catch {
        console.warn('[setup] Could not install wd_parallel. TP/SP training may be unavailable.');
        console.warn(`[setup] To fix manually, run: pip install -e "${pkgPath}"\n`);
    }
}

function runSetup() {
    ensureCudaDirectBackend();
    ensureWdParallel();
}

module.exports = { runSetup };
