if ~ exist('resultArray', 'var')
    fprintf('resultArray does not exist. run at least one test file first.\n');
end


for r = resultArray
    fprintf('writing file %s:\n', r.name);
    writeSignal(r.name, [r.time ; r.X]);
end