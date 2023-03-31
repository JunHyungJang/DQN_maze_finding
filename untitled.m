pe = pyenv;
if pe.Status == 'Loaded'
    disp('To change the Python version, restart MATLAB, then call pyenv('Version','2.7').')
else
    pyenv('Version','2.7');
end