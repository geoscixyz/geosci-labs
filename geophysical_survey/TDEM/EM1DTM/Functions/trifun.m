function out = trifun(time, ta, tb)
% function trifun(t, ta, tb)

out  = zeros(numel(time), 1);
out(time<=ta) = 1/ta*time(time<=ta);
out(time>ta & time <tb) = -1./(tb-ta)*(time(time>ta & time <tb) - tb);
