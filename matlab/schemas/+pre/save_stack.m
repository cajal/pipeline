function save_stack( key, outfile)
    X = squeeze(pre.SegmentMask.load_scan(key, 1));
    save(outfile, 'X','-v7.3');
end

