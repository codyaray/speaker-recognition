function check_identified(value, threshold, name)

if value < 1-threshold,
    disp([name '     identified! ' num2str(value)]);
else
    disp([name ' not identified! ' num2str(value)]);
end

end