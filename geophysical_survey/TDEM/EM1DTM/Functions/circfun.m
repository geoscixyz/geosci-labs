function [x, y] = circfun(xc, yc, r, n)
theta = linspace(-pi, pi, n+1);
x = r*cos(theta(:))+xc;
y = r*sin(theta(:))+yc;

x = x(end-1:-1:1);
y = y(end-1:-1:1);
end