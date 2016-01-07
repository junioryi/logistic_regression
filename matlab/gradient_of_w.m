function [ gw ] = gradient_of_w( M, labels, w, C )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

ywx = dot(M, w, 1);
disp(M);
disp(ywx);
gw = w;
end

