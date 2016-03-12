function tuning_summary
% summarizes the quality of mice
set(groot,'defaultAxesColorOrder', parula(4)*0.7, 'defaultAxesLineStyleOrder', '-|--|:')
von = aod_monet.VonMises & 'preprocess_id=3' & 'spike_inference=2';
keys = fetch(vis2p.Experiments & aod_monet.VonMises, 'exp_date', 'ORDER BY exp_date DESC');
scores = arrayfun(@(key) fetch(von & key, 'von_base->w1','von_amp1->w2','von_amp2->w3','exp(-von_sharp)->r'), keys, 'uni', false);
scores = cellfun(@(s) osifun(s), scores, 'uni', false);
mx = max(cellfun(@(s)  quantile(s, 0.98), scores));
mn = 0.0;
bins = linspace(mn, mx, 100);
h = cellfun(@(s) cumsum(hist(s,bins)), scores, 'uni', false);
h = cat(1,h{:});


plot(bins, bsxfun(@minus, h(:,end), h)', 'LineWidth', 1);
xlabel OSI
title 'number of "cells" above OSI'
grid on 
box off
xlim([mn mx])
legend(arrayfun(@(key) sprintf('%d  (%s)',key.mouse_id, key.exp_date), keys, 'uni', false))
legend Location EastOutside
legend boxoff
hold on
plot([1 1]*0.4, ylim, 'r-')
hold off

set(gcf,'PaperSize',[5 2.5], 'PaperPosition', [0 0 5 2.5])
print('-dpdf',getLocalPath('~/Desktop/quality-tuning'))


plot(bins, 100-100*bsxfun(@rdivide, h, h(:,end))', 'LineWidth', 1);
xlabel OSI
title 'percent "cells" above OSI'
grid on 
box off
xlim([mn mx])
legend(arrayfun(@(key) sprintf('%d (%s)',key.mouse_id, key.exp_date), keys, 'uni', false))
legend Location EastOutside
legend boxoff
hold on
plot([1 1]*0.4, ylim, 'r-')
hold off

set(gcf,'PaperSize',[5 2.5], 'PaperPosition', [0 0 5 2.5])
print('-dpdf',getLocalPath('~/Desktop/quality-tuning-percent'))

end


function osi = osifun(s) 
numerator = (1-[s.r]).*([s.w2] - [s.w3].*[s.r]);
denominator = 2*[s.w1] + (1+[s.r]).*([s.w2] + [s.w3].*[s.r]);
osi = numerator./denominator;
end
    