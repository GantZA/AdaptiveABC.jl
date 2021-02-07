# Plot Recipe Adapted from gpABC.jl

function dist_range(dist::T where T <: Distribution)
      return minimum(dist), maximum(dist)
end


@recipe function f(abc_out::ABCRejOutput; iteration_colours=nothing,
      iterations=nothing, params_true=nothing, prior_dists=nothing, param_inds=nothing, plot_size=(1200, 800),
      param_names=nothing
      )
      legend --> false

      layout := size(param_inds, 1)^ 2
      margin := 3mm
      size := plot_size
      posterior_mean = parameter_means(abc_out)[param_inds, :]
      posterior_median = median(abc_out.parameters, dims=2)[param_inds, :]

      if param_names === nothing
            param_names = abc_out.parameter_names[param_inds]
      end

      for (i, par1) in enumerate(param_names)
            for (j, par2) in enumerate(param_names)
                  subplot := (i - 1) * size(param_inds, 1) + j
                  if i == j
                        @series begin
                              seriestype := :histogram
                              yguide := "Density"
                              xguide := "$(par1)"
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              bins := 20
                              normalize := true
                              data = abc_out.parameters[param_inds[i], :]
                        end  # @series
                        @series begin
                              seriestype := :line
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              linewidth := 4
                              seriescolor := "blue"
                              data = (range(dist_range(prior_dists[i])..., length=10), [pdf(prior_dists[i], range(dist_range(prior_dists[i])..., length=10))])
                        end  # @series
                        @series begin
                              seriestype := :vline
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              linewidth := 4
                              seriescolor := "red"
                              data = [params_true[i]]
                        end  # @series
                        @series begin
                              seriestype := :vline
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              linewidth := 4
                              seriescolor := "green"
                              data = [posterior_mean[i]]
                        end  # @series
                        @series begin
                              seriestype := :vline
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              yguide := "Density"
                              xguide := "$(par1)"
                              linewidth := 4
                              seriescolor := "black"
                              data = [posterior_median[i]]
                        end  # @series
                  elseif j < i
                        seriestype := :scatter
                        markerstrokecolor --> false
                        for iter in iterations
                              @series begin
                                    xguide --> "$(par2)"
                                    yguide --> "$(par1)"
                                    xlims := dist_range(prior_dists[j]).+(-0.05, 0.1)
                                    ylims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                                    seriescolor := iteration_colours[iter]
                                    data = (abc_out.parameters[param_inds[j], :], abc_out.parameters[param_inds[i], :])
                              end # @series
                        end # for iter
                        @series begin
                              xguide --> "$(par2)"
                              yguide --> "$(par1)"
                              xlims := dist_range(prior_dists[j]).+(-0.05, 0.1)
                              ylims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              seriescolor := "red"
                              data = ([params_true[j]], [params_true[i]])
                        end # @series
                  elseif (i==1) & (j==size(param_inds, 1))
                        @series begin
                              legend := true
                              seriestype := :vline
                              seriescolor := "blue"
                              label := "Prior"
                              linewidth := 4
                              data = []
                        end # @series
                        @series begin
                              legend := true
                              seriestype := :vline
                              seriescolor := "red"
                              label := "True Parameter Value"
                              linewidth := 4
                              data = []
                        end # @series
                        @series begin
                              legend := true
                              seriestype := :vline
                              seriescolor := "green"
                              label := "Mean"
                              linewidth := 4
                              data = []
                        end # @series
                        @series begin
                              yguide := ""
                              xguide := ""
                              grid := false
                              xaxis := false
                              yaxis := false
                              legend := true
                              seriestype := :vline
                              seriescolor := "black"
                              label := "Median"
                              linewidth := 4
                              data = []
                        end # @series
                  else
                        @series begin
                              yguide := ""
                              xguide := ""
                              grid := false
                              xaxis := false
                              yaxis := false
                              data = []
                        end # @series
                  end # if/elseif
            end # for j, par2
      end # for i, par1
end



@recipe function f(abc_out::ABCPMCOutput; iteration_colours=nothing, iterations=nothing, 
      params_true=nothing, prior_dists=nothing, param_inds=nothing, plot_size=(1200, 800), 
      param_names=nothing
      )
      legend --> false
      layout := size(param_inds, 1)^ 2
      margin := 3mm
      size := plot_size
      posterior_mean = parameter_means(abc_out)[param_inds, iterations[end]]
      posterior_median = median(abc_out.parameters[param_inds, :, iterations[end]], dims=2)

      if param_names === nothing
            param_names = abc_out.parameter_names[param_inds]
      end

      for (i, par1) in enumerate(param_names)
            for (j, par2) in enumerate(param_names)
                  subplot := (i - 1) * size(param_inds, 1) + j
                  if i == j
                        @series begin
                              seriestype := :histogram
                              yguide := "Density"
                              xguide := "$(par1)"
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              bins := 20
                              normalize := true
                              data = abc_out.parameters[param_inds[i], :, iterations[end]]
                        end  # @series
                        @series begin
                              seriestype := :line
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              linewidth := 4
                              seriescolor := "blue"
                              data = (range(dist_range(prior_dists[i])..., length=10), [pdf(prior_dists[i], range(dist_range(prior_dists[i])..., length=10))])
                        end  # @series
                        @series begin
                              seriestype := :vline
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              linewidth := 4
                              seriescolor := "red"
                              data = [params_true[i]]
                        end  # @series
                        @series begin
                              seriestype := :vline
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              linewidth := 4
                              seriescolor := "green"
                              data = [posterior_mean[i]]
                        end  # @series
                        @series begin
                              seriestype := :vline
                              xlims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              yguide := "Density"
                              xguide := "$(par1)"
                              linewidth := 4
                              seriescolor := "black"
                              data = [posterior_median[i]]
                        end  # @series
                  elseif j < i
                        seriestype := :scatter
                        markerstrokecolor --> false
                        for (k, iter) in enumerate(iterations)
                              @series begin
                                    xguide --> "$(par2)"
                                    yguide --> "$(par1)"
                                    seriescolor := iteration_colours[k]
                                    data = (abc_out.parameters[param_inds[j], :, iter], abc_out.parameters[param_inds[i], :, iter])
                              end # @series
                        end # for iter
                        @series begin
                              xguide --> "$(par2)"
                              yguide --> "$(par1)"
                              xlims := dist_range(prior_dists[j]).+(-0.05, 0.1)
                              ylims := dist_range(prior_dists[i]).+(-0.05, 0.1)
                              seriescolor := "red"
                              data = ([params_true[j]], [params_true[i]])
                        end # @series
                  elseif (i==1) & (j==size(param_inds, 1))
                        @series begin
                              legend := true
                              seriestype := :vline
                              seriescolor := "blue"
                              label := "Prior"
                              linewidth := 4
                              data = []
                        end # @series
                        @series begin
                              legend := true
                              seriestype := :vline
                              seriescolor := "red"
                              label := "True Parameter Value"
                              linewidth := 4
                              data = []
                        end # @series
                        @series begin
                              legend := true
                              seriestype := :vline
                              seriescolor := "green"
                              label := "Mean"
                              linewidth := 4
                              data = []
                        end # @series
                        @series begin
                              yguide := ""
                              xguide := ""
                              grid := false
                              xaxis := false
                              yaxis := false
                              legend := true
                              seriestype := :vline
                              seriescolor := "black"
                              label := "Median"
                              linewidth := 4
                              data = []
                        end # @series
                  else
                        @series begin
                              yguide := ""
                              xguide := ""
                              grid := false
                              xaxis := false
                              yaxis := false
                              data = []
                        end # @series
                  end # if/elseif
            end # for j, par2
      end # for i, par1
end

