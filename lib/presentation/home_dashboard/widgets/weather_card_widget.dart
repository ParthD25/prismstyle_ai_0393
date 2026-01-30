import 'package:flutter/material.dart';
import 'package:sizer/sizer.dart';

class WeatherCardWidget extends StatelessWidget {
  final Map<String, dynamic> weatherData;

  const WeatherCardWidget({super.key, required this.weatherData});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final condition = weatherData["condition"] as String;
    final gradientColors = _getWeatherGradient(condition);

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: gradientColors,
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: gradientColors[0].withValues(alpha: 0.4),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Stack(
        children: [
          // Decorative background elements
          Positioned(
            right: -20,
            top: -20,
            child: Container(
              width: 40.w,
              height: 40.w,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withValues(alpha: 0.1),
              ),
            ),
          ),
          Positioned(
            right: 15.w,
            bottom: -10,
            child: Container(
              width: 25.w,
              height: 25.w,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withValues(alpha: 0.08),
              ),
            ),
          ),

          // Main content
          Padding(
            padding: EdgeInsets.all(5.w),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Location row
                Row(
                  children: [
                    Icon(
                      Icons.location_on_rounded,
                      color: Colors.white.withValues(alpha: 0.9),
                      size: 18,
                    ),
                    SizedBox(width: 1.w),
                    Expanded(
                      child: Text(
                        weatherData["location"] as String,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: Colors.white.withValues(alpha: 0.9),
                          fontWeight: FontWeight.w500,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),

                SizedBox(height: 2.h),

                // Temperature and icon row
                Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    // Temperature
                    Expanded(
                      flex: 2,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            weatherData["temperature"] as String,
                            style: TextStyle(
                              fontSize: 48.sp,
                              fontWeight: FontWeight.w300,
                              color: Colors.white,
                              height: 1,
                            ),
                          ),
                          SizedBox(height: 0.5.h),
                          Text(
                            condition,
                            style: theme.textTheme.titleMedium?.copyWith(
                              color: Colors.white.withValues(alpha: 0.85),
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),

                    // Weather icon (large and beautiful)
                    Container(
                      width: 22.w,
                      height: 22.w,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Center(child: _buildWeatherAnimation(condition)),
                    ),
                  ],
                ),

                SizedBox(height: 2.h),

                // Outfit recommendation
                Container(
                  padding: EdgeInsets.symmetric(
                    horizontal: 3.w,
                    vertical: 1.2.h,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.checkroom_rounded,
                        color: Colors.white,
                        size: 18,
                      ),
                      SizedBox(width: 2.w),
                      Flexible(
                        child: Text(
                          weatherData["appropriateness"] as String,
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: Colors.white,
                            fontWeight: FontWeight.w500,
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildWeatherAnimation(String condition) {
    final lowerCondition = condition.toLowerCase();
    IconData icon;

    if (lowerCondition.contains('sun') || lowerCondition.contains('clear')) {
      icon = Icons.wb_sunny_rounded;
    } else if (lowerCondition.contains('partly')) {
      icon = Icons.cloud_queue_rounded;
    } else if (lowerCondition.contains('cloud') ||
        lowerCondition.contains('overcast')) {
      icon = Icons.cloud_rounded;
    } else if (lowerCondition.contains('rain') ||
        lowerCondition.contains('drizzle')) {
      icon = Icons.water_drop_rounded;
    } else if (lowerCondition.contains('snow') ||
        lowerCondition.contains('sleet')) {
      icon = Icons.ac_unit_rounded;
    } else if (lowerCondition.contains('thunder') ||
        lowerCondition.contains('storm')) {
      icon = Icons.thunderstorm_rounded;
    } else if (lowerCondition.contains('fog') ||
        lowerCondition.contains('mist')) {
      icon = Icons.foggy;
    } else if (lowerCondition.contains('wind')) {
      icon = Icons.air_rounded;
    } else {
      icon = Icons.cloud_rounded;
    }

    return Icon(icon, size: 48, color: Colors.white);
  }

  List<Color> _getWeatherGradient(String condition) {
    final lowerCondition = condition.toLowerCase();

    if (lowerCondition.contains('sun') || lowerCondition.contains('clear')) {
      return [const Color(0xFFFF9500), const Color(0xFFFF5E3A)];
    } else if (lowerCondition.contains('partly')) {
      return [const Color(0xFF5AC8FA), const Color(0xFF007AFF)];
    } else if (lowerCondition.contains('cloud') ||
        lowerCondition.contains('overcast')) {
      return [const Color(0xFF8E8E93), const Color(0xFF636366)];
    } else if (lowerCondition.contains('rain') ||
        lowerCondition.contains('drizzle')) {
      return [const Color(0xFF5856D6), const Color(0xFF007AFF)];
    } else if (lowerCondition.contains('snow') ||
        lowerCondition.contains('sleet')) {
      return [const Color(0xFFAFD8F8), const Color(0xFF5AC8FA)];
    } else if (lowerCondition.contains('thunder') ||
        lowerCondition.contains('storm')) {
      return [const Color(0xFF5856D6), const Color(0xFF30276B)];
    } else if (lowerCondition.contains('fog') ||
        lowerCondition.contains('mist')) {
      return [const Color(0xFFAEAEB2), const Color(0xFF8E8E93)];
    } else if (lowerCondition.contains('wind')) {
      return [const Color(0xFF64D2FF), const Color(0xFF5AC8FA)];
    }
    return [const Color(0xFF5AC8FA), const Color(0xFF007AFF)];
  }
}
