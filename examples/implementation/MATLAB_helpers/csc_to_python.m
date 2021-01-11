function [succeeded, data_array, time_base, time_offset ] = csc_to_python(csc_name)

 [succeeded, data_array, time_stamps, ~, ~, ~, ~, ~ ] = NlxGetNewCSCData(csc_name);
 
 if succeeded && ~isempty(time_stamps) 
 
     time_base = int64(time_stamps(1));
     time_offset = int32(time_stamps - time_stamps(1)  );
     
 else
      time_base = nan;
  time_offset = nan;
  succeeded = false;
 end
end

