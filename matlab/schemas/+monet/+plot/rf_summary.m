function rf_summary
% summarizes the quality of mice
set(groot,'defaultAxesColorOrder', parula(4)*0.7, 'defaultAxesLineStyleOrder', '-|--|:')
fits = pro(monet.Fit, 'radius*pow(contrast,1.5)->score') & 'segment_method=2' & 'spike_inference=2';
keys = fetch(pro(common.Animal, rf.Session, 'max(session_date)->sess_date') & monet.Fit, 'sess_date', 'ORDER BY sess_date DESC');
scores = arrayfun(@(key) fetchn(fits & key, 'score'), keys, 'uni', false);
mx = max(cellfun(@(s)  quantile(s, 0.98), scores));
mn = 0.0;
bins = linspace(mn, mx, 100);
h = cellfun(@(s) cumsum(hist(s,bins)), scores, 'uni', false);
h = cat(1,h{:});


plot(bins, bsxfun(@minus, h(:,end), h)', 'LineWidth', 1);
xlabel 'RF score'
title 'number of "cells" above RF score'
grid on 
box off
xlim([mn mx])
legend(arrayfun(@(key) sprintf('%d  (%s)',key.animal_id, key.sess_date), keys, 'uni', false))
legend Location EastOutside
legend boxoff
hold on
plot([1 1]*0.04, ylim, 'r-')
hold off

set(gcf,'PaperSize',[5 2.5], 'PaperPosition', [0 0 5 2.5])
print('-dpdf',getLocalPath('~/Desktop/quality-rfs'))


plot(bins, 100-100*bsxfun(@rdivide, h, h(:,end))', 'LineWidth', 1);
xlabel 'RF score'
title 'percent "cells" above RF score'
grid on 
box off
xlim([mn mx])
legend(arrayfun(@(key) sprintf('%d (%s)',key.animal_id, key.sess_date), keys, 'uni', false))
legend Location EastOutside
legend boxoff
hold on
plot([1 1]*0.04, ylim, 'r-')
hold off

set(gcf,'PaperSize',[5 2.5], 'PaperPosition', [0 0 5 2.5])
print('-dpdf',getLocalPath('~/Desktop/quality-rfs-percent'))


end