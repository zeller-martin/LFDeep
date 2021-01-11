function [time] = get_teensy_timestamp(sync_bit)

time_received = false;

while ~time_received

    [succeeded, time_stamps, ~, ttl_values, ~, ~, ~ ] = NlxGetNewEventData('Events');
    if succeeded

        for i = length(ttl_values):-1:1
          if bitget(ttl_values(i), sync_bit)
            time = time_stamps(i);
            time_received = true;
            break
          end
        end
    end
end


end
