if ~ exist('resultArray', 'var')
    fprintf('resultArray does not exist. run at least one test file first.\n');
end


for r = resultFile
    fprintf('writing file %s:\n', r.name);
    writeFile(r.name, [r.time ; r.X]);
end