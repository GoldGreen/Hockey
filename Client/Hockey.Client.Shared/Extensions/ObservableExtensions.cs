using System;
using System.Collections.Generic;

namespace Hockey.Client.Shared.Extensions
{
    public static class ObservableExtensions
    {
        private static List<IDisposable> isposables = new();

        public static IDisposable Cashe(this IDisposable disposable)
        {
            isposables.Add(disposable);
            return disposable;
        }
    }
}
