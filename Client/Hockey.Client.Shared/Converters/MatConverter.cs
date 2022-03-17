using OpenCvSharp;
using System;
using System.Globalization;

namespace Hockey.Client.Shared.Converters
{
    public class MatConverter : ConverterBase<MatConverter>
    {
        public override object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is Mat m && m != null && !m.IsDisposed && !m.Empty())
            {
                return m.ToBytes();
            }

            return null;
        }
    }
}
