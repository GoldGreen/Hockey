using Microsoft.Extensions.Configuration;
using Prism.Ioc;
using System.IO;

namespace Hockey.Client.Shared.Configuration
{
    public static class ConfigurationExtensions
    {
        public static IContainerRegistry RegisterConfiguration(this IContainerRegistry containerProvider)
        {
            containerProvider.RegisterSingleton<IConfiguration>(CreateConfiguration);
            return containerProvider;
        }

        public static IContainerRegistry RegisterOption<T>(this IContainerRegistry containerProvider, string section)
        {
            containerProvider.RegisterSingleton<T>(c => c.OptionFactory<T>(section));
            return containerProvider;
        }

        private static IConfiguration CreateConfiguration()
        {
            return new ConfigurationBuilder()
                       .SetBasePath(Directory.GetCurrentDirectory())
                       .AddJsonFile("appsettings.json")
                       .Build();
        }

        private static T OptionFactory<T>(this IContainerProvider containerProvider, string section)
        {
            return containerProvider.Resolve<IConfiguration>()
                                    .GetSection(section)
                                    .Get<T>();
        }
    }
}
