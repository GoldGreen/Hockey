using Hockey.Server.BusinessLayer.Configuration;
using Hockey.Server.BusinessLayer.Services.Abstractions;
using Hockey.Server.BusinessLayer.Services.Implementation;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace Hockey.Server.BusinessLayer
{
    public static class BusinessLayerExtensions
    {
        public static IServiceCollection AddBusinessLayer(this IServiceCollection services, IConfiguration configuration)
        {
            services.Configure<PythonInitOption>(configuration.GetSection(PythonInitOption.PythonInit));

            services.AddSingleton<IHockeyService, HockeyService>()
                    .AddSingleton<IPythonInitService, PythonInitService>();

            return services;
        }
    }
}
