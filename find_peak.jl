# distinguish the shape of distribution according to number of peaks
# prob0: the probability distribution
# type: the type of distribution
function find_peak(prob0)
      diff1 = sign.(diff(vec(prob0)))
      Lmax = diff(diff1) .== -2
      L0 = ifelse(diff1[1]<0,1,0)
      Lmax = [L0;Lmax;0]
      peak_numb = sum(Lmax)
      if peak_numb==1&&L0==1
            type = 1
      elseif peak_numb==1&&L0==0
            type = 2
      elseif peak_numb==2&&L0==1
            type = 3
      else peak_numb==2&&L0==0
            type = 4
      end
      return(type)
end
